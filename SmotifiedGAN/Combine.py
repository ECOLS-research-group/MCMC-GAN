import DataProcessing as DataProcessing

option = int(input("Enter 1 for ecoli\nEnter 2 for poker\nEnter 3 for winequality\nEnter 4 for yeast\nEnter 5 for abalone\nEnter 6 for ionosphere\nEnter 7 for spambase\nEnter 8 for page block\nEnter 9 for CreditCard\nEnter 10 for COVID\nEnter 11 for Breast Cancer\n"))

if option == 1:
    DataProcessing.EcoliProcessing()
elif option ==2:
    DataProcessing.PokerProcessing()
elif option ==3:
    DataProcessing.WineQualityProcessing()
elif option ==4:
    DataProcessing.YeastProcessing()
elif option ==5:
    DataProcessing.AbaloneProcessing()
elif option ==6:
    DataProcessing.IonosphereProcessing()
elif option ==7:
    DataProcessing.SpambaseProcessing()
elif option ==8:
    DataProcessing.PageBlockProcessing()
elif option ==9:
    DataProcessing.CreditCard()
elif option ==10:
    DataProcessing.covid()
elif option ==11:
    DataProcessing.breastCancer()
 


