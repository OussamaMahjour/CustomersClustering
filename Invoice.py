from pandas import DataFrame


class Invoice (DataFrame):
    def __init__(self,InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country) -> None:
        self.InvoiceNo = InvoiceNo
        self.StockCode = StockCode
        self.Quantity = Quantity
        self.Description = Description
        self.InvoiceDate = InvoiceDate,
        self.UnitPrice = UnitPrice
        self.CustomerID	= CustomerID
        self.Country = Country
