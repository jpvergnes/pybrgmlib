"""
Extract ADES time series
"""
import re
import os
import psycopg2
import pandas as pd


def requete_pz_chroniques(codes, filtre='', only_point_eau=False):
    """
    Request time series of piezometers from the ADES Clone. Write the file
    point_eau.csv with the description of the piezometers.
    Write the time series in csv in a specific folder.

    Parameters
    ----------
    codes : list
        BSS codes to extract
    filtre : string, optional
        Filter to apply on the database. Default to ''
    only_point_eau : bool, optional
        If True, only write the "point_eau.csv" file. Default to False.
    """
    ades = BDAdes()
    codes = ades.filter_codes(filtre, codes)
    df_dic = ades.request_point_eau(codes)
    df_dic.to_csv('point_eau.csv', sep=';', index=False)
    if not only_point_eau:
        ades.write_pz_chronique(codes)
    ades.write_log()


class BDAdes():
    """
    Class BDAdes
    """
    def __init__(self):
        """
        Initialization of a BDAdes object
        """
        self.log = []
        # self.f = open('requete.log', 'w')
        conn = psycopg2.connect(
            dbname='ades',
            user='user_r',
            host='10.100.1.229',
            password='ruser'
        )
        self.cur = conn.cursor()

        self.cur.execute("""SELECT * FROM information_schema.columns
                WHERE table_schema = 'ades' AND table_name = 'point_eau'""")
        self.names_pt_eau = [row[3] for row in self.cur.fetchall()]

        self.cur.execute("""SELECT * FROM information_schema.columns
                WHERE table_schema = 'ades' AND table_name = 'pz_chronique'""")
        self.names_pz_chronique = [row[3] for row in self.cur.fetchall()]

    def filter_codes(self, filtre, codes):
        """
        Apply a filter on the codes list

        Parameters
        ----------
        filtre : string, optional
            Filter to apply on the database. Default to ''
        codes : list
            BSS codes to extract

        Returns
        -------
        out : list
            A list of filtered codes
        """
        codes = list(codes)
        codes_filtre = []
        indice_bss = self.names_pt_eau.index('code_bss')
        if filtre:
            self.log.append("Filtre détecté !")
            self.cur.execute(
                """SELECT * FROM ades.point_eau WHERE {0}""".format(filtre)
            )
            rows = self.cur.fetchall()
            codes_filtre = [row[indice_bss] for row in rows]
        if codes and codes_filtre:
            codes2 = []
            for code in codes:
                if code in codes_filtre:
                    self.log.append("{0} dans le filtre".format(code))
                    codes2.append(code)
                else:
                    self.log.append("{0} pas dans le filtre".format(code))
            codes = codes2
        elif codes_filtre:
            codes = codes_filtre
        return codes

    def request_point_eau(self, codes):
        """
        Request the point_eau table

        Parameters
        ----------
        codes : list
            BSS codes to request

        Returns
        -------
        out : pandas.DataFrame
            A pandas.DataFrame object describing the requested piezometers
        """
        self.log.append("I - Sélection dans la table point_eau")
        dic = {name: [] for name in self.names_pt_eau}
        for bss in codes:
            self.cur.execute("""SELECT * FROM ades.point_eau
                    WHERE code_bss LIKE '%{0}%'""".format(
                        bss.upper()))
            rows = self.cur.fetchone()
            if (
                    (re.findall('^[0-9]', bss) and
                     rows and
                     rows[self.names_pt_eau.index('nb_mesures_piezo')] > 0)
            ):
                code_bss = rows[self.names_pt_eau.index('code_bss')]
                self.log.append(
                    "{0} trouvé dans la table point_eau".format(code_bss)
                )
                for i, row in enumerate(rows):
                    dic[self.names_pt_eau[i]].append(row)
            else:
                self.log.append(
                    "{0} absent de la table point_eau".format(bss)
                )
        return pd.DataFrame(dic, columns=self.names_pt_eau)

    def write_pz_chronique(self, codes):
        """
        Write time series in csv files

        Parameters
        ----------
        codes : list
            BSS codes to request
        """
        self.log.append("II - Extraction des chroniques piézométriques")
        for code_bss in codes:
            self.cur.execute("""SELECT * FROM ades.pz_chronique
                    WHERE code_bss LIKE '%{0}%'""".format(
                        code_bss))
            rows = self.cur.fetchall()
            if not rows:
                self.log.append(
                    '{0} absent de la table pz_chronique'.format(code_bss)
                )
                continue
            df_pz = pd.DataFrame(rows, columns=self.names_pz_chronique)
            df_pz = df_pz.set_index(pd.to_datetime(df_pz['date_mesure']))
            df_pz = df_pz.sort_index()
            if not os.path.isdir("pz_chronique"):
                os.makedirs("pz_chronique")
            df_pz.to_csv(
                'pz_chronique/{0}.csv'.format(rows[0][0].replace('/', '_')),
                sep=';', encoding='cp1252'
                )

    def write_log(self):
        """
        Write the log
        """
        with open('request.log', 'w') as flog:
            flog.write('\n'.join(self.log))
