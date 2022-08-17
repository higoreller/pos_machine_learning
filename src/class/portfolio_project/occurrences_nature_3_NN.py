import numpy
import pandas
import tensorflow
from tensorflow import keras
from matplotlib import pyplot
import seaborn
import itertools
import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, r2_score
from tensorflow.python.client import device_lib

tensorflow.config.list_physical_devices('GPU')

# DATA PRE-PREPARATION
file_path = '../../../dataset/occurrences.xlsx'

df = pandas.read_excel(file_path)
df = df.drop(["rai", "obm_afeto", "qualificacao", "sexo"], axis=1) 

#REMOVING NULL VALUES
df.loc[pandas.isnull(df["data"])]
df.loc[pandas.isnull(df["naturezas"])]
df.loc[pandas.isnull(df["bairro_cidade"])]
df.loc[pandas.isnull(df["tr"])]
df.loc[pandas.isnull(df["obm_escala"])]

df = df.loc[df["bairro_cidade"] != "(null)"]
df = df.loc[df["recurso"] != "(null)"]
df = df.loc[df["tr"] != "(null)"]
df = df.loc[df["obm_escala"] != "(null)"]

#TRANSFORMING "data" COLUMN INTO NEW COLUMNS "dia" e "periodo". ALSO TRANSFORMING "tr" COLUMN INTO "tempo_resposta" COLUMN

def day_name(timestamp):
    weekdays = ('Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo')
    return weekdays[timestamp.weekday()]

def period_of_day(timestamp):
    period = ("Madrugada", "Matutino", "Vespertino", "Noturno")
    # Madrugada 00:00 às 05:59
    # Matutino 06:00 às 11:59
    # Vespertino 12:00 às 17:59
    # Noturno 18:00 às 23:59
    if timestamp.hour >= 0 and timestamp.hour < 6:
        return period[0]
    elif timestamp.hour >= 6 and timestamp.hour < 12:
        return period[1]
    elif timestamp.hour >= 12 and timestamp.hour < 18:
        return period[2]
    elif timestamp.hour >= 18 and timestamp.hour < 24:
        return period[3]



def response_time(response_time):
    # Muito rápido 0 a 10 minutos
    # Rápido 10 a 15 minutos
    # Médio 15 a 20 minutos
    # Longo 20 a 30 minutos
    # Muito longo 30 a 45 minutos
    # Extremamente longo > 45 minutos

    response_time_metric = ("Muito rápido", "Rápido", "Médio", "Longo", "Muito longo", "Extremamente longo")

    if type(response_time) is datetime.time:

        total_time_in_minutes = response_time.hour*60 + response_time.minute + response_time.second/60

        if total_time_in_minutes >= 0 and total_time_in_minutes <= 10:
            return response_time_metric[0]
        elif total_time_in_minutes > 10 and total_time_in_minutes <= 15:
            return response_time_metric[1]
        elif total_time_in_minutes > 15 and total_time_in_minutes <= 20:
            return response_time_metric[2]
        elif total_time_in_minutes > 20 and total_time_in_minutes <= 30:
            return response_time_metric[3]
        elif total_time_in_minutes > 30 and total_time_in_minutes <= 45:
            return response_time_metric[4]
        elif total_time_in_minutes > 45:
            return response_time_metric[5]
        
#Lembrar de remover os valores que não são datetime.time do df["tr"]
df.loc[:, "dia"] = df["data"].apply(day_name)
df.loc[:, "periodo"] = df["data"].apply(period_of_day)
df.loc[:, "tempo_resposta"] = df["tr"].apply(response_time)

#REMOVING "DATA" AND "TR" COLUMNS

df = df.drop(["data", "tr"], axis=1)

#Removing None values
df = df.dropna()
df = df.mask(df.eq('None')).dropna()
df = df.astype(str)