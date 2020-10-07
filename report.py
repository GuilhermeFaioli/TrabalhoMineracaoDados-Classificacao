"""
Created on Sat Sep 26 18:29:15 2020
@author: Fabiana Barreto Pereira, Guilherme Ferreira Faioli Lima, Junior Reis dos Santos, Rafaela Silva Miranda e Yasmine de Melo Leite
"""

from datetime import datetime
#from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class Report:

    def __init__(self,fileName):
        self.doc = SimpleDocTemplate(fileName,pagesize=A4,
                        rightMargin=52,leftMargin=52,
                        topMargin=52,bottomMargin=18)
        self.styles = getSampleStyleSheet()
        self.numberColumns = 0
        self.numberRows = 0
        self.numberRowsTrain = 0
        self.numberRowsTest = 0
        self.nNeighbors = []

    '''
    Método adiciona novos estilos de parágrafo
    '''
    def defineStyles(self):
        self.styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, fontName ='Times-Roman', fontSize = 12))


    def setNumberColumnsAndRows(self,nColTotal,nRowTotal,nRowTrain,nRowTest):
        self.numberColumns = nColTotal
        self.numberRows = nRowTotal
        self.numberRowsTrain = nRowTrain
        self.numberRowsTest = nRowTest



    '''
    Método gera relatório no formato pdf
    '''
    def generateReport(self,graphics,neighbors):
        self.defineStyles()
        head = self.head()
        self.nNeighbors = neighbors
        graphics = self.graphics(graphics)

        story = head + graphics

        self.doc.build(story)

    '''
    Método adiciona informações básicas ao relatório
    '''
    def head(self):
        story = []
        texto = '<font name=Times-Roman>Sistemas de Apoio a Decisão</font>'
        story.append(Paragraph(texto,self.styles['Title']))
        story.append(Spacer(1, 12))
        texto = '<font name=Times-Roman>CLASSIFICAÇÃO</font>'
        story.append(Paragraph(texto,self.styles['Title']))
        story.append(Spacer(1, 12))
        texto = '<font size=16 name=Times-Roman>BASE DE DADOS: SRAG - 2020</font>'
        story.append(Paragraph(texto,self.styles['Title']))
        story.append(Spacer(1, 12))
        texto = '<font name=Times-Bold>Descrição da base de dados: </font>\
        <font>Ficha individual de casos de síndrome respiratória aguda grave hospitalizados</font>'
        story.append(Paragraph(texto,self.styles['Justify']))
        story.append(Spacer(1, 12))
        texto = '<font size=12 name=Times-Bold>Tipo de classificação: </font>\
        <font>Diagnóstico de SRAG (1 - SRAG por influenza, 2 - SRAG por outro vírus respiratório, 3 - SRAG por outro agente etiológico,\
         4 - SRAG não especificado, 5 - SRAG por COVID-19)</font>'
        story.append(Paragraph(texto,self.styles['Justify']))
        story.append(Spacer(1, 12))
        texto = '<font name=Times-Bold>Equipe: </font><font>Fabiana Barreto Pereira, Guilherme Ferreira Faioli Lima,\
         Junior Reis dos Santos, Rafaela Silva Miranda e Yasmine de Melo Leite</font>'
        story.append(Paragraph(texto,self.styles['Justify']))
        story.append(Spacer(1, 12))
        texto = '<font name=Times-Bold>Discente: </font><font>Janniele Aparecida Soares Araujo</font>'
        story.append(Paragraph(texto,self.styles['Justify']))
        story.append(Spacer(1, 12))
        data = datetime.now().strftime('%d/%m/%Y - %H:%M')
        texto = '<font name=Times-Bold>Data da análise: </font><font>'+data+'</font>'
        story.append(Paragraph(texto,self.styles['Justify']))
        story.append(Spacer(1, 12))
        texto = '<font name=Times-Bold>Total de linhas da base de dados: </font><font>725615</font>'
        story.append(Paragraph(texto,self.styles['Justify']))
        story.append(Spacer(1, 12))
        texto = '<font name=Times-Bold>Total de colunas da base de dados: </font><font>154</font>'
        story.append(Paragraph(texto,self.styles['Justify']))
        story.append(Spacer(1, 12))
        texto = '<font name=Times-Bold>Total de linhas consideradas: </font><font>'+str(self.numberRows)+' ('+str(self.numberRowsTrain)+' para treinamento e '+str(self.numberRowsTest)+' para teste)</font>'
        story.append(Paragraph(texto,self.styles['Justify']))
        story.append(Spacer(1, 12))
        texto = '<font name=Times-Bold>Total de colunas consideradas: </font><font>'+str(self.numberColumns)+'</font>'
        story.append(Paragraph(texto,self.styles['Justify']))
        story.append(Spacer(1, 12))
        texto = '<font name=Times-Bold>Algoritmo: </font><font>K-NN (KNeighborsClassifier da biblioteca Scikit-learn)</font>'
        story.append(Paragraph(texto,self.styles['Justify']))
        story.append(Spacer(1, 12))

        return story


    def graphics(self,graphics):
        story = []
        story.append(Spacer(1, 20))
        # Balanceamento do banco
        texto = '<font fontSize=12>Verificando balanceamento do conjunto de dados</font>'
        story.append(Paragraph(texto,self.styles['Title']))
        story.append(Spacer(1, 12))

        image = graphics[0]
        im = Image(image, self.in2p(4), self.in2p(3))
        story.append(im)
        story.append(Spacer(1, 12))
        # precisão
        texto = '<font fontSize=12>Precisão dos dados de teste</font>'
        story.append(Paragraph(texto,self.styles['Title']))
        story.append(Spacer(1, 12))

        image = graphics[len(graphics)-1]
        im = Image(image, self.in2p(5), self.in2p(2))
        story.append(im)
        story.append(Spacer(1, 12))
        #análise para cada k

        texto = '<font fontSize=12>Avaliação por número de vizinhos (k)</font>'
        story.append(Paragraph(texto,self.styles['Title']))
        story.append(Spacer(1, 12))

        for k in self.nNeighbors:
            texto = '<font fontSize=12>Avaliação para k = '+str(k)+'</font>'
            story.append(Paragraph(texto,self.styles['Title']))
            story.append(Spacer(1, 12))

            for i in range(1,len(graphics)-1):
                test = "_"+str(k)
                if test in graphics[i]:
                    image = graphics[i]
                    if 'Erro' in graphics[i]:
                        im = Image(image, self.in2p(5), self.in2p(2))
                    else:
                        im = Image(image, self.in2p(4), self.in2p(3))
                    story.append(im)

            story.append(Spacer(1, 20))


        return story


    '''
    Método transforma Inch em pt
    '''
    def in2p(self,Inch):
        return Inch *72


if __name__ == "__main__":
    r = Report("Relatório.pdf")
    r.generateReport()
