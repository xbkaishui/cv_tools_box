from loguru import logger
import datetime


def test_parse_data():
    DepthMarketData = {
        "TradingDay": b"20240228",
        "reserve1": b"",
        "ExchangeID": b"",
        "reserve2": b"",
        "LastPrice": 5270.6,
        "PreSettlementPrice": 5311.8,
        "PreClosePrice": 5355.8,
        "PreOpenInterest": 13393.0,
        "OpenPrice": 5367.2,
        "HighestPrice": 5429.2,
        "LowestPrice": 5211.8,
        "Volume": 8033,
        "Turnover": 8562972560.0,
        "OpenInterest": 14483.0,
        "ClosePrice": 1.7976931348623157e308,
        "SettlementPrice": 1.7976931348623157e308,
        "UpperLimitPrice": 5842.8,
        "LowerLimitPrice": 4780.8,
        "PreDelta": 1.7976931348623157e308,
        "CurrDelta": 1.7976931348623157e308,
        "UpdateTime": b"13:18:23",
        "UpdateMillisec": 500,
        "BidPrice1": 5270.8,
        "BidVolume1": 1,
        "AskPrice1": 5272.2,
        "AskVolume1": 1,
        "BidPrice2": 1.7976931348623157e308,
        "BidVolume2": 0,
        "AskPrice2": 1.7976931348623157e308,
        "AskVolume2": 0,
        "BidPrice3": 1.7976931348623157e308,
        "BidVolume3": 0,
        "AskPrice3": 1.7976931348623157e308,
        "AskVolume3": 0,
        "BidPrice4": 1.7976931348623157e308,
        "BidVolume4": 0,
        "AskPrice4": 1.7976931348623157e308,
        "AskVolume4": 0,
        "BidPrice5": 1.7976931348623157e308,
        "BidVolume5": 0,
        "AskPrice5": 1.7976931348623157e308,
        "AskVolume5": 0,
        "AveragePrice": 1065974.4254948338,
        "ActionDay": b"20240228",
        "InstrumentID": b"IM2404",
        "ExchangeInstID": b"",
        "BandingUpperPrice": 0.0,
        "BandingLowerPrice": 0.0,
    }
    time = datetime.datetime.strptime(
        str(DepthMarketData["ActionDay"] + DepthMarketData["UpdateTime"], "gb2312"),
        "%Y%m%d%H:%M:%S",
    ).replace(microsecond=DepthMarketData["UpdateMillisec"] * 1000)
    open = DepthMarketData["OpenPrice"]
    close = DepthMarketData["ClosePrice"]
    pre_close = DepthMarketData["PreClosePrice"]
    low = DepthMarketData["LowestPrice"]
    high = DepthMarketData["HighestPrice"]
    logger.info(
        "open {}, close {}, pre_close {}, low {}, high {}",
        open,
        close,
        pre_close,
        low,
        high,
    )
    logger.info("time {}", str(time))


def query_quote_IM():
    url = "http://www.cffex.com.cn/quote_IM.txt"
    # TODO use requests get data from url
    import requests
    import pandas as pd
    import io

    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Referer": "http://www.cffex.com.cn/zz1000/",
        },
    )
    text_value = response.text
    # read str to df
    df = pd.read_csv(io.StringIO(text_value))
    print(df)
    instrument_ids = df['instrument'].tolist()
    instrument_id_bytes = [instrument_id.encode('utf-8') for instrument_id in instrument_ids]
    logger.info("instrument_id_bytes {}", instrument_id_bytes)


if __name__ == "__main__":
    # test_parse_data()
    query_quote_IM()
