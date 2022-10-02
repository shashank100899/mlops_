import flask_sqlalchemy
import pandas 

df = pandas.read_csv("winequality-red.csv",sep=";")

def test_check_columnLength():
    assert len(df.columns) == 12

def test_check_columnType():
    check = True
    for i in df.dtypes:
        if i == float or i == int:
            pass
        else:
            check = False
            break
    assert check
        
