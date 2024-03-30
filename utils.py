class TernaryOperator():
    @staticmethod
    def operate(a: bool, b, c):
        return b if a is True else c
    
    @staticmethod
    def operate_elif(a: bool, b, c: bool, d, e):
        return b if a is True else (d if c is True else e)

