import os
import datetime
import chardet
from calendar import monthrange
import pandas as pd
import numpy as np

def set_df(df, line):
    fields = [field for field in line.strip().split(';') if field]
    if len(fields) == 2:
        df[fields[0].strip()] = fields[1].strip()
    elif len(fields) > 2:
        df[fields[0]] = fields[1:]
    else:
        df[fields[0]] = ''
    return df

class FicheStation(object):
    def __init__(self, fichier):
        rawdata = open(fichier, 'rb').read()
        encoding = chardet.detect(rawdata)['encoding']
        self.f = open(fichier, 'r', encoding=encoding)
        self.line = self.f.readline()
    
    def read_line(self):
        self.line = self.f.readline()
    
    def read_block(self, line):
        df = {}
        while line.strip():
            df = set_df(df, line)
            line = self.f.readline()
            self.line = line
        return df

    def extract_block(self):
        while not self.line.strip():
            self.read_line()
        return self.read_block(self.line)

    def extract_table(self):
        while not self.line.strip():
            self.read_line()
        return self.read_table(self.line)

    def read_table(self, line):
        dcoord = set_df({}, line)
        self.read_line()
        fields = self.line.strip().split(';')
        self.read_line()
        df = {}
        i = 0
        while self.line.strip():
            values = self.line.strip().split(';')
            if not df:
                df = {field.strip():value.strip() for field, value in zip(fields, values) if field}
            else:
                df.update({'{0}.{1}'.format(field.strip(), i):value.strip()
                for field, value in zip(fields, values) if field})
            i += 1
            self.read_line()
        dcoord.update(df)
        return dcoord


class QJM(FicheStation):
    def extract_table(self):
        for line in self.f:
            if 'Procédure' not in line and line.strip():
                fields = line.strip().split(';')
                line = self.f.readline()
                values = line.strip().split(';')
                df = {field.strip():value.strip() for field, value in zip(fields, values) if field}
                self.line = line
                return df

    def extract_year(self):
        for line in self.f:
            if line.strip():
                dyear = set_df({}, line)
                self.line = line
                return dyear
    
    def extract_debits_mensuels(self, annee):
        for line in self.f:
            if 'Débits mensuels' in line:
                line = self.f.readline()
                fields = line.strip().split(';')[1:-2]
                values = []
                for _ in range(12):
                    line = self.f.readline()
                    values.append(line.strip().split(';')[:-3])
                values = np.array(values)
                df = pd.DataFrame(values[:, 1:], index=values[:, 0], columns=fields)
                df = df.replace('', -2)
                df = df.replace('\xa0', -2)
                df = df.astype('float')
                start = datetime.datetime(annee, 1, 1)
                end = datetime.datetime(annee, 12, 31)
                date_range = pd.date_range(start, end, freq='MS')
                self.line = line
                return df.set_index(date_range)

    def pass_statistiques(self):
        for line in self.f:
            if 'Statistiques' in line:
                break
        while not line.strip():
            self.line = line
            self.f.readline()

    @classmethod
    def build_final(cls, data, annee):
        final = []
        for i in range(0, 12):
            toto = data[31*i:31*(i+1)]
            nbdays = monthrange(annee, i+1)[1]
            final.append(toto[:nbdays])
        final = np.concatenate(final, axis=0)
        start = datetime.datetime(annee, 1, 1)
        end = datetime.datetime(annee, 12, 31)
        date_range = pd.date_range(start, end)
        return pd.Series(final, index=date_range)

    def extract_debits_journaliers(self, annee):
        for line in self.f:
            if "Débits journaliers" in line:
                for _ in range(2):
                    self.f.readline()
                data = []
                for i in range(31):
                    line = self.f.readline().strip().split(';')
                    line = [-2 if i == '' else float(i) for i in line[1:-1][::2]]
                    data.append(line)
                data = np.array(data)
                data = data.T.flatten()
                self.line = self.f.readline()
                return self.build_final(data, annee)


def read_station_form(fichier):
    """
    Read station form file in .csv downloaded from BDHydro2

    Parameter
    ---------
    fichier : str

    Return
    ------
    pandas.DataFrame
    """
    fs = FicheStation(fichier)
    df = {}
    while fs.line:
        fs.read_line()
        sr = fs.extract_block() # Code station
        sr.update(fs.extract_block()) # Paramètres descriptifs
        sr.update(fs.extract_block()) # Données hydros
        sr.update(fs.extract_table()) # Altitude du zéro de l'échelle
        while 'Coordonnées' not in fs.line:
            fs.read_line()
        sr.update(fs.extract_table()) # Coordonnées
        sr.update(fs.extract_block()) # Stations remplacées
        dsr = fs.extract_block()
        dsr = {'Débits {0}'.format(field):value for field, value in dsr.items()}
        sr.update(dsr) # Données débits dispos
        dsr = fs.extract_block()
        dsr = {'Hauteurs {0}'.format(field):value for field, value in dsr.items()}
        sr.update(dsr) # Données hauteurs dispos
        df[sr['Code station']] = pd.Series(sr)
    fs.f.close()
    return pd.DataFrame(df).T

def read_time_series(fichier):
    """
    Read qjm file in .csv downloaded from BDHydro2

    Parameter
    ---------
    fichier : str

    Return
    ------
    pandas.DataFrame
    """
    qjm = QJM(fichier)
    df = {}
    while qjm.line:
        dstations = qjm.extract_table() # Stations
        dtemp = df.get(dstations['Code station'], pd.Series([]))
        dyear = qjm.extract_year()
        _ = qjm.extract_debits_mensuels(int(dyear['Année']))
        qjm.pass_statistiques()
        dfqjm = qjm.extract_debits_journaliers(int(dyear['Année']))
        df[dstations['Code station']] = pd.concat([dtemp, dfqjm], axis=0)
    df = pd.DataFrame(df)
    df.index.set_names('Date', inplace=True)
    return pd.DataFrame(df)

def request_bdh(qjm_file=None, station_file=None):
    """
    Extract time series from qjm_file or station_file

    Parameters
    ----------
    qjm_file : str (optional)
    station_file : str (optional)
    """
    f = open('request.log', 'w')
    if station_file:
        df = read_station_form(station_file)
        df.to_csv('fiche_station.csv', sep=';', encoding='cp1252', index=False)
        f.write('{0} written in fiche_station.csv\n'.format(station_file))
    if qjm_file:
        dfq = read_time_series(qjm_file)
        codes = dfq.columns
        for code in codes:
            assert code in dfq.columns
            if not os.path.isdir("qjm_chronique"):
                os.makedirs("qjm_chronique")
            dfq.loc[:, code].to_csv(
                'qjm_chronique/{0}.csv'.format(code),
                sep=';', encoding='cp1252'
            )
            f.write(
                '{0} found in {1} and written '
                'in qjm_chronique/{0}.csv\n'.format(code, qjm_file)
                )
    f.close()
               