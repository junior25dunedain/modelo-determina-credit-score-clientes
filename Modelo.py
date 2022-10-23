import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import r2_score # a melhor metrica para avaliar um modelo regressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder # usado para fazer o onehotencoding
from sklearn.linear_model import LinearRegression # modelo regressor linear


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

# carregando os dados dos clientes
df = pd.read_excel('dados_credito.xlsx')

#analise exploratoria dos dados
print(df.shape)
print(df.sample(5))

# a variavel CODIGO_CLIENTE não é importante para essa aplicação e será removida
# as variaveis UF,ESCOLARIDADE,ESTADO_CIVIL,CASA_PROPRIA,OUTRA_RENDA,TRABALHANDO_ATUALMENTE
# vão passar por um tratamento onehotencoding(trnasformar dados categoricos em numericos)
# a variavel ULTIMO_SALARIO terá seus dados convertidos para o tipo float64
print(df.info())


##tratando os dados
# removendo CODIGO_CLIENTE
df.drop('CODIGO_CLIENTE',axis=1,inplace=True)

print(df.groupby('ULTIMO_SALARIO').size(),'\n')

df.replace('SEM DADOS',np.nan,inplace=True)

df['ULTIMO_SALARIO'] = df['ULTIMO_SALARIO'].astype('float64')

print(f"A media dos dados de ultimo salario é {df['ULTIMO_SALARIO'].mean()}")
print(f"A mediana dos dados de ultimo salario é {df['ULTIMO_SALARIO'].median()}")
print(f"A moda dos dados de ultimo salario é {df['ULTIMO_SALARIO'].mode()}")

# possui 3 dados missing em ULTIMO_SALARIO
print(df.isna().sum(),'\n')

df['ULTIMO_SALARIO'].fillna(df['ULTIMO_SALARIO'].median(),inplace=True)
print(df.info(),'\n')

# observando as variaveis numericas
print(df.describe())

var_numericas = []
for i in df.columns:
    if df.dtypes[i] == 'int64' or df.dtypes[i] == 'float64':
        print(i,':',df.dtypes[i])
        var_numericas.append(i)

print(f'quantidade de variaveis numericas: {len(var_numericas)}')


# criando uma figura com todos os boxplots das variaveis numericas
# para verificar a ocorrencia de outliers
plt.rcParams['figure.figsize'] = [15.00,12.00]
plt.rcParams['figure.autolayout'] = True

f, axe = plt.subplots(2,5)
linha = 0
coluna = 0

for i in var_numericas:
    sns.boxplot(data= df,y = i,ax=axe[linha][coluna])
    coluna +=1
    if coluna == 5:
        linha += 1
        coluna = 0
plt.show()

# Os outliers encontrados estão na variavel QT_FILHOS
print(df.loc[df['QT_FILHOS']>5])
# removendo os outliers
df.drop(df.loc[df['QT_FILHOS']>5].index,inplace=True)

# criando uma figura com todos os histogramas das variaveis numericas

plt.rcParams['figure.figsize'] = [15.00,12.00]
plt.rcParams['figure.autolayout'] = True

f, axe = plt.subplots(4,3)
linha = 0
coluna = 0

for i in var_numericas:
    sns.histplot(data= df,x = i,ax=axe[linha][coluna])
    coluna +=1
    if coluna == 3:
        linha += 1
        coluna = 0
plt.show()

# criando um hitmap para observar a correlação entre as variaveis do dataframe
plt.rcParams['figure.figsize'] = (18,8)
ax = sns.heatmap(df.corr(),annot=True)
plt.show()


# grafico de dispersão para a correlação de algumas variaveis
sns.lmplot(x='ULTIMO_SALARIO',y='SCORE',data=df)
plt.show()
# grafico de dispersão para a correlação de algumas variaveis
sns.lmplot(x='TEMPO_ULTIMO_EMPREGO_MESES',y='SCORE',data=df)
plt.show()
# grafico de dispersão para a correlação de algumas variaveis
sns.lmplot(x='VL_IMOVEIS',y='SCORE',data=df)
plt.show()

# fazendo uma engenharia de dados, criando uma nova variavel faixa etaria
print(f'menor idade: {df.IDADE.min()}')
print(f'maior idade: {df.IDADE.max()}','\n')

idades = [0,30,40,50,60]
nomes = ['Até 30','31 a 40','41 a 50','Maior que 50']
df['FAIXA_ETARIA'] = pd.cut(df.IDADE,bins=idades,labels=nomes)
print(df['FAIXA_ETARIA'].value_counts(),'\n')

# observando a media do score para cada grupo de faixa etaria
print(df.groupby(['FAIXA_ETARIA']).mean()['SCORE'])

# acessando as variaveis categoricas
var_categoricas = []
for j in df.columns:
    if df.dtypes[j] == 'object' or df.dtypes[j] == 'category':
        print(j,':',df.dtypes[j])
        var_categoricas.append(j)

print(f'numero de variaveis categoricas: {len(var_categoricas)}')


# criando uma figura com todos os countplot das variaveis categoricas
plt.rcParams['figure.figsize'] = [15.00,22.00]
plt.rcParams['figure.autolayout'] = True

f, axe = plt.subplots(4,2)
linha = 0
coluna = 0

for i in var_categoricas:
    sns.countplot(data= df,x = i,ax=axe[linha][coluna])
    coluna +=1
    if coluna == 2:
        linha += 1
        coluna = 0
plt.show()

## pre-processamento dos dados
# criando encoder
lb = LabelEncoder()

df['FAIXA_ETARIA'] = lb.fit_transform(df['FAIXA_ETARIA'])
df['TRABALHANDO_ATUALMENTE'] = lb.fit_transform(df['TRABALHANDO_ATUALMENTE'])
df['OUTRA_RENDA'] = lb.fit_transform(df['OUTRA_RENDA'])
df['CASA_PROPRIA'] = lb.fit_transform(df['CASA_PROPRIA'])
df['ESTADO_CIVIL'] = lb.fit_transform(df['ESTADO_CIVIL'])
df['ESCOLARIDADE'] = lb.fit_transform(df['ESCOLARIDADE'])
df['UF'] = lb.fit_transform(df['UF'])

# limpando os dados NAN
df.dropna(inplace=True)

print(df.info())

# separando dados de entrada e saida
alvo = df['SCORE'].copy()
entradas = df.drop('SCORE',axis=1)

# separando dados de treino e teste
x_treino,x_teste,y_treino,y_teste = train_test_split(entradas,alvo,test_size=0.3,random_state=40)

# realizando a normalização dos dados de entrada
normalizador  = MinMaxScaler()

x_treino_norm = normalizador.fit_transform(x_treino)
x_teste_norm = normalizador.fit_transform(x_teste)

## modelo regressor linear
modelo = LinearRegression(normalize=True)
modelo.fit(x_treino_norm,y_treino)

print(r2_score(y_teste,modelo.predict(x_teste_norm)))


# aplicação de novos dados ao modelo preditor
UF = 5
IDADE = 26
ESCOLARIDADE = 2
ESTADO_CIVIL = 1
QT_FILHOS = 0
CASA_PROPRIA = 2
QT_IMOVEIS = 2
VL_IMOVEIS = 350000
OUTRA_RENDA = 1
OUTRA_RENDA_VALOR = 2500
TEMPO_ULTIMO_EMPREGO_MESES = 20
TRABALHANDO_ATUALMENTE = 1
ULTIMO_SALARIO = 6500.00
QT_CARROS = 2
VALOR_TABELA_CARROS = 75000
FAIXA_ETARIA = 1

novos = [UF,IDADE,ESCOLARIDADE,ESTADO_CIVIL,QT_FILHOS,CASA_PROPRIA,QT_IMOVEIS,VL_IMOVEIS,
         OUTRA_RENDA,OUTRA_RENDA_VALOR,TEMPO_ULTIMO_EMPREGO_MESES,TRABALHANDO_ATUALMENTE,ULTIMO_SALARIO,
         QT_CARROS,VALOR_TABELA_CARROS,FAIXA_ETARIA]

X = np.array(novos).reshape(1,-1)
X = normalizador.transform(X)

print(f'O score de credito previsto para esse cliente é {modelo.predict(X)}')

