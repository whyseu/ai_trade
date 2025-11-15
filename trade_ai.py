import json
import math
import time
from datetime import datetime

import ccxt
import pandas as pd
import requests
from openai import OpenAI

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key='xxxxxx',
    base_url="https://api.deepseek.com"
)

# 初始化OKX交易所
exchange = ccxt.okx({
    'options': {
        'defaultType': 'swap',  # OKX使用swap表示永续合约
    },
    'apiKey': "xxxxxx",
    'secret': "xxxxxx",
    'password': "xxxxxx",  # OKX需要交易密码
})
# 预加载OKX市场数据，确保后续可用`exchange.market()`访问交易对信息
exchange.load_markets()

# 交易参数配置 - 支持多币种和多时间框架
TRADE_CONFIG = {
    'coins': ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'BNB'],  # 支持的币种列表
    'quote': 'USDT',  # 计价货币
    'contract_type': ':USDT',  # OKX永续合约后缀
    'leverage_range': [5, 10, 20, 30, 50, 75],  # 杠杆范围
    'timeframes': {
        'intraday': '15m',  # 日内数据（15分钟）
        'background': '4h',  # 背景数据（4小时）
        'macro': '1d',  # 宏观趋势（日线）
    },
    'data_points': {
        'intraday': 50,  # 日内数据点数量（增加以确保EMA/RSI/ATR有足够预热期）
        'background': 100,  # 背景数据点数量（增加以确保EMA50/SMA50有足够预热期）
        'macro': 30,  # 日线数据点数量
    },
    'decision_interval': 900,  # 决策频率（秒），15分钟 = 900秒
    'test_mode': False,  # 测试模式
    'initial_capital': 1000,  # 起始资金（美元）
    'margin_mode': 'isolated',  # OKX 保证金模式（cross 或 isolated）
}

start_time = datetime.now()  # 记录开始时间
coin_data = {}  # 每个币种数据
max_profit_history = {}  # 每个币种的历史最大盈利（百分比）
current_step = 1
sharpe_ratio = 0
signal_history = []

def get_symbol_for_coin(coin):
    """根据币种生成OKX交易对符号"""
    if "/" in coin or ":" in coin:
        return coin
    return f"{coin}/{TRADE_CONFIG['quote']}{TRADE_CONFIG['contract_type']}"


def fetch_funding_rate(symbol):
    """获取资金费率"""
    funding_data = exchange.fetch_funding_rate(symbol)
    return {
        'rate': float(funding_data.get('fundingRate', 0)),
        'timestamp': funding_data.get('timestamp', None),
        'next_funding_time': funding_data.get('fundingTimestamp', None)
    }


def fetch_orderbook_imbalance(symbol):
    """获取订单簿失衡度 - 预测短期方向"""
    try:
        orderbook = exchange.fetch_order_book(symbol, limit=20)

        # 计算买单和卖单总量
        bid_volume = sum([bid[1] for bid in orderbook['bids'][:20]])
        ask_volume = sum([ask[1] for ask in orderbook['asks'][:20]])

        # 计算失衡度：(买单量 - 卖单量) / (买单量 + 卖单量)
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            imbalance = (bid_volume - ask_volume) / total_volume
        else:
            imbalance = 0

        # 解读
        if imbalance > 0.3:
            interpretation = "多头优势"
        elif imbalance < -0.3:
            interpretation = "空头优势"
        else:
            interpretation = "均衡"

        return {
            'value': imbalance,
            'interpretation': interpretation,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume
        }
    except Exception as e:
        print(f"获取订单簿失衡失败: {e}")
        return {
            'value': 0,
            'interpretation': "数据不可用",
            'bid_volume': 0,
            'ask_volume': 0
        }


def calculate_vwap(df):
    """计算VWAP（成交量加权平均价格）- 判断价格位置"""
    # 典型价格 = (high + low + close) / 3
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

    # VWAP = cumsum(典型价格 × volume) / cumsum(volume)
    df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()

    # 清理临时列
    df.drop(['typical_price'], axis=1, inplace=True)

    return df


# 计算技术指标 - 整合原有指标和system.md要求的指标
def calculate_technical_indicators(df):
    # 移动平均线 (SMA)
    df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()

    # 指数移动平均线 (EMA) - 使用标准递归EMA计算
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()  # 新增：system.md要求
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()  # 新增：system.md要求

    # MACD指标（Moving Average Convergence Divergence）
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()  # Signal线也需要adjust=False
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # 相对强弱指数 (RSI) - 使用Wilder平滑方法（标准RSI计算）
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # RSI 7周期 - 使用指数加权移动平均（EMA），alpha=1/7
    avg_gain_7 = gain.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    avg_loss_7 = loss.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    rs_7 = avg_gain_7 / avg_loss_7
    df['rsi_7'] = 100 - (100 / (1 + rs_7))

    # RSI 14周期 - 使用指数加权移动平均（EMA），alpha=1/14
    avg_gain_14 = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss_14 = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs_14 = avg_gain_14 / avg_loss_14
    df['rsi_14'] = 100 - (100 / (1 + rs_14))
    df['rsi'] = df['rsi_14']  # 向后兼容

    # ATR（平均真实波幅）- 新增：system.md要求
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    # 使用指数加权移动平均计算ATR（Wilder's smoothing: alpha=1/period）
    df['atr_3'] = df['true_range'].ewm(alpha=1 / 3, min_periods=3, adjust=False).mean()  # 3周期ATR
    df['atr_14'] = df['true_range'].ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()  # 14周期ATR

    # 清理临时���
    df.drop(['high_low', 'high_close', 'low_close', 'true_range'], axis=1, inplace=True)

    # 布林带（Bollinger Bands）
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    # 计算布林带位置，添加除以零保护
    bb_width = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / bb_width).where(bb_width > 0, 0.5)  # 宽度为0时默认0.5（中间位置）

    # 成交量均线
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # 支撑阻力位
    df['resistance'] = df['high'].rolling(20).max()
    df['support'] = df['low'].rolling(20).min()

    # VWAP（成交量加权平均价格）
    df = calculate_vwap(df)

    # 填充NaN值
    df = df.bfill().ffill()

    return df


def get_multi_timeframe_data(coin):
    """获取多时间框架数据（15分钟日内 + 4小时背景 + 日线宏观）"""
    symbol = get_symbol_for_coin(coin)

    try:
        # 获取15分钟日内数据
        ohlcv_intraday = exchange.fetch_ohlcv(
            symbol,
            TRADE_CONFIG['timeframes']['intraday'],
            limit=TRADE_CONFIG['data_points']['intraday']
        )
        df_intraday = pd.DataFrame(ohlcv_intraday, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_intraday['timestamp'] = pd.to_datetime(df_intraday['timestamp'], unit='ms')
        df_intraday = calculate_technical_indicators(df_intraday)

        # 获取4小时背景数据
        ohlcv_4h = exchange.fetch_ohlcv(
            symbol,
            TRADE_CONFIG['timeframes']['background'],
            limit=TRADE_CONFIG['data_points']['background']
        )
        df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')
        df_4h = calculate_technical_indicators(df_4h)

        # 获取日线宏观数据
        ohlcv_1d = exchange.fetch_ohlcv(
            symbol,
            TRADE_CONFIG['timeframes']['macro'],
            limit=TRADE_CONFIG['data_points']['macro']
        )
        df_1d = pd.DataFrame(ohlcv_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'], unit='ms')
        df_1d = calculate_technical_indicators(df_1d)

        # 获取永续合约数据
        funding_data = fetch_funding_rate(symbol)

        # 获取新增指标数据
        orderbook_imbalance = fetch_orderbook_imbalance(symbol)
        # 当前数据快照
        current_intraday = df_intraday.iloc[-1]
        current_4h = df_4h.iloc[-1]
        current_1d = df_1d.iloc[-1]
        current_price = float(current_intraday['close'])

        # --- 多时间框架趋势判断 ---
        # 15分钟趋势
        price_15m = float(current_intraday['close'])
        ema20_15m = float(current_intraday['ema_20'])
        ema50_15m = float(current_intraday['ema_50'])

        trend_15m = "Sideways"
        if price_15m > ema20_15m and ema20_15m > ema50_15m:
            trend_15m = "Uptrend"
        elif price_15m < ema20_15m and ema20_15m < ema50_15m:
            trend_15m = "Downtrend"

        # 4小时趋势
        price_4h = float(current_4h['close'])
        ema20_4h = float(current_4h['ema_20'])
        ema50_4h = float(current_4h['ema_50'])
        atr14_4h = float(current_4h['atr_14'])

        trend_4h = "Sideways"
        if price_4h > ema20_4h and ema20_4h > ema50_4h:
            trend_4h = "Uptrend"
        elif price_4h < ema20_4h and ema20_4h < ema50_4h:
            trend_4h = "Downtrend"

        # 日线趋势
        price_1d = float(current_1d['close'])
        ema20_1d = float(current_1d['ema_20'])
        ema50_1d = float(current_1d['ema_50'])

        trend_1d = "Sideways"
        if price_1d > ema20_1d and ema20_1d > ema50_1d:
            trend_1d = "Uptrend"
        elif price_1d < ema20_1d and ema20_1d < ema50_1d:
            trend_1d = "Downtrend"

        # 波动性判断（基于4小时ATR）
        atr_percentage = (atr14_4h / price_4h) * 100 if price_4h > 0 else 0
        volatility = "High" if atr_percentage > 4 else "Low"
        # --- 结束 ---

        # 获取市场限制信息（最小交易数量等）
        market = exchange.market(symbol)
        min_amount = float(market['limits']['amount']['min'])  # 最小合约张数
        amount_precision = market['precision']['amount']  # 张数精度
        contract_size = float(market.get('contractSize', 1))  # 每张合约代表的币数量

        # 计算最小USD要求：最小张数 × 每张合约币数 × 当前价格
        min_usd = min_amount * contract_size * current_price

        return {
            'coin': coin,
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'volatility': volatility,

            # 多时间框架趋势
            'trend_15m': trend_15m,
            'trend_4h': trend_4h,
            'trend_1d': trend_1d,

            # 市场限制信息（本系统以USD为交易单位）
            'limits': {
                'min_amount': min_amount,  # 最小合约张数
                'amount_precision': amount_precision,  # 张数精度
                'contract_size': contract_size,  # 每张合约对应的币数量
                'min_usd': min_usd,  # 最小交易USD金额
            },

            # 日内数据（15分钟）
            'intraday': {
                'timeframe': TRADE_CONFIG['timeframes']['intraday'],
                'prices': df_intraday['close'].tolist(),
                'ema_20': df_intraday['ema_20'].tolist(),
                'macd': df_intraday['macd'].tolist(),
                'rsi_7': df_intraday['rsi_7'].tolist(),
                'rsi_14': df_intraday['rsi_14'].tolist(),
                'current': {
                    'price': current_price,
                    'ema_20': float(current_intraday['ema_20']),
                    'macd': float(current_intraday['macd']),
                    'rsi_7': float(current_intraday['rsi_7']),
                    'rsi_14': float(current_intraday['rsi_14']),
                    'vwap': float(current_intraday['vwap']),
                }
            },

            # 背景数据（4小时）
            'background': {
                'timeframe': TRADE_CONFIG['timeframes']['background'],
                'ema_20': ema20_4h,
                'ema_50': ema50_4h,
                'atr_3': float(current_4h['atr_3']),
                'atr_14': atr14_4h,
                'volume_current': float(current_4h['volume']),
                'volume_avg': float(current_4h['volume_ma']),
                'macd': df_4h['macd'].tolist(),
                'rsi_14': df_4h['rsi_14'].tolist(),
            },

            # 宏观数据（日线）
            'macro': {
                'timeframe': TRADE_CONFIG['timeframes']['macro'],
                'trend': trend_1d,
                'ema_20': ema20_1d,
                'ema_50': ema50_1d,
                'rsi_14': float(current_1d['rsi_14']),
                'macd': float(current_1d['macd']),
            },

            # 永续合约数据
            'perpetual': {
                'funding_rate': funding_data['rate'],
                'orderbook_imbalance': orderbook_imbalance,
            },

            # 完整DataFrame供进一步分析
            'df_intraday': df_intraday,
            'df_4h': df_4h,
            'df_1d': df_1d,
        }

    except Exception as e:
        print(f"获取 {coin} 数据失败: {e}")
        return None


def collect_all_coins_data():
    """收集所有币种的数据"""
    all_coins_data = {}

    for coin in TRADE_CONFIG['coins']:
        print(f"正在获取 {coin} 数据...")
        data = get_multi_timeframe_data(coin)
        if data:
            all_coins_data[coin] = data
        else:
            print(f"⚠️ {coin} 数据获取失败，跳过")
    return all_coins_data


def get_current_position(symbol=None):
    """获取当前持仓情况 - 支持多币种"""
    try:
        if symbol:
            # 获取特定币种的持仓
            positions = exchange.fetch_positions([symbol])
        else:
            # 获取所有持仓
            positions = exchange.fetch_positions()

        result = []
        for pos in positions:
            contracts = float(pos.get('contracts', 0) or 0)

            lastUpdateTimestamp = float(pos.get('lastUpdateTimestamp')) / 1000
            # 毫秒时间戳转时间
            minute = math.ceil((datetime.now().timestamp() - lastUpdateTimestamp) / 60)

            # OKX 对冲模式下，检查 info 中的持仓数量
            if contracts == 0 and pos.get('info'):
                pos_info = pos['info']
                contracts = float(pos_info.get('pos', 0) or 0)

            if contracts > 0:
                # 从 info 中获取更准确的持仓方向
                pos_side = pos.get('side')  # ccxt 标准字段
                if pos.get('info'):
                    # OKX 原始数据中的 posSide 字段
                    pos_side = pos['info'].get('posSide', pos_side)

                # 获取合约信息以计算准确的持仓价值
                market = exchange.market(pos['symbol'])
                contract_size = float(market.get('contractSize', 1))  # 每张合约对应的币数量

                current_price = float(pos.get('markPrice', 0) or 0)

                result.append({
                    'coin': pos['symbol'].split('/')[0],  # 提取币种名称
                    'symbol': pos['symbol'],
                    'side': pos_side,  # 'long' or 'short'
                    'quantity': contracts,  # 合约张数（非币种数量）
                    'contract_size': contract_size,  # 每张合约对应的币数量
                    'entry_price': float(pos['entryPrice']) if pos.get('entryPrice') else 0,  # USD价格
                    'current_price': current_price,  # USD价格
                    'liquidation_price': float(pos.get('liquidationPrice', 0) or 0),  # USD价格
                    'unrealized_pnl': float(pos.get('unrealizedPnl', 0) or 0),  # USD盈亏
                    'leverage': float(pos.get('leverage', 1) or 1),
                    'margin_mode': pos.get('info', {}).get('mgnMode', 'cross') if pos.get('info') else 'cross',
                    'durationMin': minute,
                    'position_value_usd': contracts * contract_size * current_price,  # 持仓价值(USD) = 张数 × 每张币数 × 价格
                })

        # 如果查询单个币种，返回单个结果或None
        if symbol:
            return result[0] if result else None

        return result

    except Exception as e:
        print(f"获取持仓失败: {e}")
        import traceback
        traceback.print_exc()
        return None if symbol else []


def calculate_sharpe_ratio(limit=30):
    """计算夏普比率 - 使用历史平仓数据API"""
    # 获取历史平仓数据（最近100次，足够计算夏普比率）
    history_data = exchange.fetch_positions_history(limit=limit)

    if not history_data or len(history_data) < 2:
        return 0.0

    # 提取每笔交易的收益率（百分比）
    returns = []
    for pos in history_data:
        if pos.get('info'):
            # pnlRatio 是小数形式（如 0.0323 表示 3.23%），转换为百分比
            pnl_ratio = float(pos['info'].get('pnlRatio', 0)) * 100
            returns.append(pnl_ratio)

    if len(returns) < 2:
        return 0.0

    # 计算平均收益率
    avg_return = sum(returns) / len(returns)

    # 计算标准差
    variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = variance ** 0.5

    if std_dev == 0:
        return 0.0

    # 计算夏普比率（假设无风险利率为0）
    sharpe = avg_return / std_dev
    return sharpe


def get_account_info():
    """获取账户信息和统计数据"""
    balance = exchange.fetch_balance()
    cash_available = float(balance['USDT']['free']) if 'USDT' in balance else TRADE_CONFIG['initial_capital']
    total_equity = float(balance['USDT']['total']) if 'USDT' in balance else cash_available

    # 获取所有持仓
    positions = get_current_position()

    # 计算总回报率
    return_pct = ((total_equity - TRADE_CONFIG['initial_capital']) / TRADE_CONFIG['initial_capital']) * 100
    # 计算已使用时间（分钟）
    minutes_elapsed = int((datetime.now() - start_time).total_seconds() / 60)

    return {
        'cash_available': cash_available,
        'account_value': total_equity,
        'return_pct': return_pct,
        'sharpe_ratio': sharpe_ratio,
        'minutes_elapsed': minutes_elapsed,
        'positions': positions,
    }


def load_system_prompt():
    """从system.md文件加载系统提示词"""
    with open('system.md', 'r', encoding='utf-8') as f:
        template = f.read()
    template = template.replace("{coin_list}", format_list_values(TRADE_CONFIG['coins']), 4)
    template = template.replace("{leverage_range}", format_list_values(TRADE_CONFIG['leverage_range']))
    template = template.replace('{decision}', f"{TRADE_CONFIG['decision_interval'] / 60}")
    template = template.replace("{current_step}", f"{current_step}")

    return template


def format_list_values(values, precision=2):
    """格式化列表值为字符串"""
    if isinstance(values, list):
        return ', '.join([f"{v:.{precision}f}" if isinstance(v, (int, float)) else str(v) for v in values])
    return str(values)


def generate_coin_technical_info(coin, data):
    """生成某个币种的技术信息字符串"""
    precision = 2
    if coin == "DOGE":
        precision = 4

    # 只显示最后10个数据点（符合system.md要求）
    def last_n(data_list, n=10):
        return data_list[-n:] if isinstance(data_list, list) and len(data_list) > n else data_list

    # 获取各时间框架的最高最低价数据
    df_intraday = data['df_intraday']
    df_4h = data['df_4h']
    df_1d = data['df_1d']
    
    # 15分钟级别最高最低价
    high_15m = float(df_intraday['high'].max()) if len(df_intraday) > 0 else 0.0
    low_15m = float(df_intraday['low'].min()) if len(df_intraday) > 0 else 0.0
    
    # 4小时级别最高最低价
    high_4h = float(df_4h['high'].max()) if len(df_4h) > 0 else 0.0
    low_4h = float(df_4h['low'].min()) if len(df_4h) > 0 else 0.0
    
    # 日线级别最高最低价
    high_1d = float(df_1d['high'].max()) if len(df_1d) > 0 else 0.0
    low_1d = float(df_1d['low'].min()) if len(df_1d) > 0 else 0.0

    info = f"""## 所有 {coin} 数据
### 当前快照
- 当前价格(USD) = ${data['current_price']:.2f}
- 市场波动性(4h) = {data['volatility']}
- 当前20周期EMA(15m) = ${data['intraday']['current']['ema_20']:.2f}
- 当前MACD(15m) = {data['intraday']['current']['macd']:.4f}
- 当前RSI(15m, 7 周期) = {data['intraday']['current']['rsi_7']:.2f}
- VWAP = ${data['intraday']['current']['vwap']:.2f}
- 资金费率：{data['perpetual']['funding_rate']:.6f}
- 订单簿失衡：{data['perpetual']['orderbook_imbalance']['interpretation']} ({data['perpetual']['orderbook_imbalance']['value']:.2f})
- 15分钟趋势：{data['trend_15m']}
- 4小时趋势：{data['trend_4h']}
- 日线趋势：{data['trend_1d']}
- 最小交易金额(保证金USD) ≥ ${data['limits']['min_usd']:.2f}

### 15分钟时间信息(最早 → 最新)
中间价：[{format_list_values(last_n(data['intraday']['prices']), precision)}]
EMA（20 周期）：[{format_list_values(last_n(data['intraday']['ema_20']), precision)}]
MACD：[{format_list_values(last_n(data['intraday']['macd']), 4)}]
RSI（7 周期）：[{format_list_values(last_n(data['intraday']['rsi_7']), precision)}]
RSI（14 周期）：[{format_list_values(last_n(data['intraday']['rsi_14']), precision)}]
最近15分钟最高价：${high_15m:.{precision}f}
最近15分钟最低价：${low_15m:.{precision}f}

## 4小时背景信息(最早 → 最新)
20周期EMA：{data['background']['ema_20']:.2f} vs. 50 周期 EMA：{data['background']['ema_50']:.2f}
3周期ATR：{data['background']['atr_3']:.2f} vs. 14 周期 ATR：{data['background']['atr_14']:.2f}
当前成交量：{data['background']['volume_current']:.2f} vs. 平均成交量：{data['background']['volume_avg']:.2f}
MACD：[{format_list_values(last_n(data['background']['macd']), 4)}]
RSI（14 周期）：[{format_list_values(last_n(data['background']['rsi_14']), precision)}]
最近4小时最高价：${high_4h:.{precision}f}
最近4小时最低价：${low_4h:.{precision}f}

## 日线宏观趋势(最早 → 最新)
趋势方向：{data['macro']['trend']}
日线20周期EMA：{data['macro']['ema_20']:.2f} vs 50周期EMA：{data['macro']['ema_50']:.4f}
日线RSI：{data['macro']['rsi_14']:.2f}，当前MACD：{data['macro']['macd']:.4f}
当天最高价：${high_1d:.{precision}f}
当天最低价：${low_1d:.{precision}f}
---"""
    return info


def format_positions_history(limit=8):
    """获取并格式化历史平仓数据"""
    # 获取最近3次平仓记录
    history_data = exchange.fetch_positions_history(limit=limit)

    if not history_data:
        return "暂无历史平仓记录"

    # 格式化历史平仓数据
    history_str = f"## 最近平仓记录参考（最近{limit}次）\n\n"

    for idx, pos in enumerate(history_data):
        # 提取重要信息
        symbol = pos.get('symbol', 'N/A').split('/')[0]  # 提取币种名称
        side = pos.get('side', 'N/A')  # long/short
        entry_price = pos.get('entryPrice', 0)  # 开仓价格
        close_price = pos.get('lastPrice', 0)  # 平仓价格
        realized_pnl = pos.get('realizedPnl', 0)  # 已实现盈亏
        leverage = pos.get('leverage', 1)  # 杠杆倍数

        # 从info中获取更详细的数据
        info = pos['info']
        pnl_ratio = float(info.get('pnlRatio', 0)) * 100  # 盈亏比率(转换为百分比)
        c_time = info.get('cTime', '0')  # '1762326176021' 毫秒时间戳
        c_time = datetime.fromtimestamp(int(c_time) / 1000).strftime('%Y-%m-%d %H:%M:%S')  # 转换为时间
        u_time = info.get('uTime', '0')  # '1762326176021' 毫秒时间戳
        u_time = datetime.fromtimestamp(int(u_time) / 1000).strftime('%Y-%m-%d %H:%M:%S')  # 转换为时间

        # 判断盈亏状态
        pnl_status = "盈利" if realized_pnl > 0 else "亏损"

        history_str += f"""- 币种:{symbol} 开仓时间: {c_time} 平仓时间: {u_time} 交易方向: {side.upper()} 开仓价格: ${entry_price:.4f} 平仓价格: ${close_price:.4f} 杠杆倍数: {leverage}x 已实现盈亏: ${realized_pnl:+.4f} ({pnl_ratio:+.2f}%) {pnl_status}\n"""

    return history_str.strip()


def generate_user_prompt(all_coins_data, account_info):
    """生成用户提示词，填充所有变量"""
    with open('user_prompt.md', 'r', encoding='utf-8') as f:
        template = f.read()

    # 替换通用变量
    current_time = datetime.now()
    running_duration = current_time - start_time

    days = running_duration.days
    hours, remainder = divmod(running_duration.seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    running_time_str = f"{days}天 {hours}小时 {minutes}分钟"

    template = template.replace('{current_date}', current_time.strftime('%Y-%m-%d %H:%M:%S'))
    template = template.replace('{running_time}', running_time_str)
    template = template.replace('{return_pct}', f"{account_info['return_pct']:.2f}")
    template = template.replace('{sharpe_ratio}', f"{account_info['sharpe_ratio']:.2f}")
    template = template.replace('{cash_available}', f"{account_info['cash_available']:.2f}")
    template = template.replace('{account_value}', f"{account_info['account_value']:.2f}")
    template = template.replace('{decision}', f"{TRADE_CONFIG['decision_interval'] / 60}")
    template = template.replace("{current_step}", f"{current_step}")

    # 获取并格式化历史平仓数据
    history_positions_str = format_positions_history()
    template = template.replace('{history_positions}', history_positions_str)

    # 生成所有币种的技术信息
    all_coins_info = ""
    for coin in TRADE_CONFIG['coins']:
        if coin in all_coins_data:
            all_coins_info += generate_coin_technical_info(coin, all_coins_data[coin])
            all_coins_info += "\n"
    template = template.replace('{all_coins_info}', all_coins_info)

    # 填充持仓信息
    if account_info['positions']:
        positions_str = "[\n"
        for pos in account_info['positions']:
            coin = pos['coin']
            # 计算盈亏百分比
            pnl_pct = 0
            if pos.get('entry_price') and pos['entry_price'] > 0:
                price_change_pct = (pos['current_price'] - pos['entry_price']) / pos['entry_price']
                if pos['side'] == 'short':
                    price_change_pct = -price_change_pct
                pnl_pct = price_change_pct * pos['leverage'] * 100

            # 更新历史最大盈利
            global max_profit_history
            if coin not in max_profit_history:
                max_profit_history[coin] = 0
            if pnl_pct > max_profit_history[coin]:
                max_profit_history[coin] = pnl_pct

            positions_str += f"""  {{
    'symbol': '{coin}',
    'side': '{pos['side']}',
    '开仓价格(USD)': {pos['entry_price']},
    '当前价格(USD)': {pos['current_price']},
    '未实现盈亏:': '{pos['unrealized_pnl']:+.2f} ({pnl_pct:+.2f}%)',
    '杠杆': {pos['leverage']}x,
    '历史最大盈利': '{max_profit_history[coin]:.2f}%',
    '退出计划': {{
      'profit_target': {coin_data.get(coin, {}).get('profit_target', '')},
      'stop_loss': {coin_data.get(coin, {}).get('stop_loss', '')},
      'invalidation_condition': '{coin_data.get(coin, {}).get('invalidation_condition', '')}',
      'confidence': '{coin_data.get(coin, {}).get('confidence', '')}',
      'justification': '{coin_data.get(coin, {}).get('justification', '')}'
    }},
    '持仓时间(分钟)': {pos['durationMin']},
    '持仓价值(美元)': {pos['position_value_usd']:.2f}
  }},
"""
        positions_str += "]"
    else:
        positions_str = "[]"
    # 替换持仓相关的占位符
    template = template.replace("{positions_str}", positions_str)
    # global signal_history
    # signal_history_str = ""
    # if len(signal_history) == 0:
    #     signal_history_str = "暂无历史判断信号"
    # else:
    #     for signal in signal_history[-3:]:
    #         signal_history_str += f"### 信号时间:{signal[0]['timestamp']}\n{json.dumps(signal,ensure_ascii=False)}\n\n"
    # template = template.replace("{signal_history}", signal_history_str)
    return template


# 抓取json
def extract_json_from_text(text):
    a = text.find('{')
    b = text.find('}', a + 1)
    c = text[a:b + 1]
    return json.loads(c)


def extract_list_from_text(text):
    a = text.find('```json')
    b = text.find('```', a + 1)
    c = text[a + 7:b].strip()
    return json.loads(c)


def analyze_with_deepseek(all_coins_data, account_info) -> list:
    """使用DeepSeek分析市场并生成交易信号 - 支持多币种"""

    # 加载系统提示词
    system_prompt = load_system_prompt()

    # 生成用户提示词
    user_prompt = generate_user_prompt(all_coins_data, account_info)
    # 将user_prompt追加写入文件
    with open('user_prompt.log', 'a+', encoding='utf-8') as f:
        if current_step == 1:
            f.write(f"系统提示词:\n{system_prompt}\n\n")
        f.write(user_prompt)
        f.write("\n\n")

    print("\n" + "=" * 60)
    print("正在调用AI分析...")
    print("=" * 60)

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False,
            temperature=0.1
        )

        # 解析JSON响应
        result = response.choices[0].message.content
        reasoning_content = response.choices[0].message.reasoning_content
        print(f"\nAI原始响应:\n{result}\n")
        text = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n思维链:\n{reasoning_content}\n\nAI回复:\n{result}"
        with open('user_prompt.log', 'a+', encoding='utf-8') as f:
            f.write("\n\n")
            f.write(text)
            f.write("\n\n")

        try:
            signal_datas = extract_list_from_text(result)
        except Exception as e:
            print(f"JSON解析错误: {e}")
            raise ValueError("Invalid JSON response")

        for index,signal_data in enumerate(signal_datas):
            # 验证必需字段（system.md格式）
            required_fields = ['signal', 'coin', 'principal', 'leverage', 'profit_target',
                               'stop_loss', 'invalidation_condition', 'confidence',
                               'justification']

            missing_fields = [field for field in required_fields if field not in signal_data]
            if missing_fields:
                print(f"⚠️ 缺少必需字段: {missing_fields}")
                raise ValueError(f"Missing required fields: {missing_fields}")

            # 验证币种
            if signal_data['coin'] not in TRADE_CONFIG['coins']:
                print(f"⚠️ 无效的币种: {signal_data['coin']}")
                raise ValueError(f"Invalid coin: {signal_data['coin']}")

            # 验证信号类型
            valid_signals = ['buy_long', 'buy_short', 'hold', 'close']
            if signal_data['signal'] not in valid_signals:
                print(f"⚠️ 无效的信号类型: {signal_data['signal']}")
                raise ValueError(f"Invalid signal: {signal_data['signal']}")

            # 保存信号到历史记录
            signal_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            signal_datas[index] = signal_data

            global coin_data
            coin_data[signal_data['coin']] = {
                'profit_target': signal_data['profit_target'],
                'stop_loss': signal_data['stop_loss'],
                'invalidation_condition': signal_data['invalidation_condition'],
                'confidence': signal_data['confidence'],
                'justification': signal_data['justification']
            }
            # 输出信号摘要
            print("\n" + "=" * 60)
            text = f"""
交易信号: {signal_data['signal']}
币种: {signal_data['coin']}
本金(USD): ${signal_data['principal']:,.2f}
杠杆: {signal_data['leverage']}x
平仓条件: {signal_data['invalidation_condition']}
信心: {signal_data['confidence']:.2f}
止盈: ${signal_data['profit_target']:,.2f}
止损: ${signal_data['stop_loss']:,.2f}
理由: {signal_data['justification']}
"""
            print(text)
            print("=" * 60)

        return signal_datas

    except Exception as e:
        print(f"AI分析失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def execute_trade(signal_data):
    """执行交易 - 支持多币种"""

    signal = signal_data['signal']
    coin = signal_data['coin']
    symbol = get_symbol_for_coin(coin)

    print("\n" + "=" * 60)
    print("交易执行")
    print("=" * 60)

    # 如果是HOLD信号，不执行任何操作
    if signal == 'hold':
        print("信号: HOLD - 保持当前状态")
        return
    # 如果是CLOSE信号，平仓
    if signal == 'close':
        close_position(symbol, signal_data)

    # 如果是BUY或SELL信号
    if signal in ['buy_long', 'buy_short']:
        # 检查信心度
        if signal_data['confidence'] < 0.3:
            print(f"⚠️ 信心度过低 ({signal_data['confidence']:.2f})，跳过交易")
            return

        # 提取止盈止损价格
        take_profit = signal_data.get('profit_target', 0)
        stop_loss = signal_data.get('stop_loss', 0)

        if signal == "buy_long":
            open_long_with_margin(symbol, signal_data['principal'], signal_data['leverage'],
                                  take_profit, stop_loss)
        elif signal == "buy_short":
            open_short_with_margin(symbol, signal_data['principal'], signal_data['leverage'],
                                   take_profit, stop_loss)
    signal_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def wait_for_next_period(period_minutes):
    """等待到下一个决策周期"""
    now = datetime.now()
    current_minute = now.minute
    current_second = now.second

    # 计算下一个决策周期的分钟数
    next_period_minute = ((current_minute // period_minutes) + 1) * period_minutes
    if next_period_minute == 60:
        next_period_minute = 0

    # 计算需要等待的总秒数
    if next_period_minute > current_minute:
        minutes_to_wait = next_period_minute - current_minute
    else:
        minutes_to_wait = 60 - current_minute + next_period_minute

    seconds_to_wait = minutes_to_wait * 60 - current_second

    # 显示友好的等待时间
    display_minutes = minutes_to_wait - 1 if current_second > 0 else minutes_to_wait
    display_seconds = 60 - current_second if current_second > 0 else 0

    if display_minutes > 0:
        print(f"🕒 等待 {display_minutes} 分 {display_seconds} 秒到下一个决策周期...")
    else:
        print(f"🕒 等待 {display_seconds} 秒到下一个决策周期...")

    return seconds_to_wait


def trading_bot():
    """主交易机器人函数 - 多币种版本"""
    print("\n" + "=" * 80)
    print(f"{'多币种AI交易系统':^80}")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        # 1. 收集所有币种数据
        print("\n[步骤 1/4] 收集市场数据...")
        all_coins_data = collect_all_coins_data()

        if not all_coins_data:
            print("❌ 无法获取任何币种数据，跳过本次决策")
            return

        print(f"✓ 成功获取 {len(all_coins_data)} 个币种的数据")

        # 显示当前价格
        print("\n当前价格:")
        for coin, data in all_coins_data.items():
            print(f"  {coin}: ${data['current_price']:,.2f}")

        # 2. 获取账户信息
        print("\n[步骤 2/4] 获取账户信息...")
        account_info = get_account_info()
        print(f"  账户价值: ${account_info['account_value']:,.2f}")
        print(f"  可用资金: ${account_info['cash_available']:,.2f}")
        print(f"  总回报率: {account_info['return_pct']:+.2f}%")
        print(f"  夏普比率: {account_info['sharpe_ratio']:.2f}")
        print(f"  持仓数量: {len(account_info['positions'])}")

        if account_info['positions']:
            print("\n  当前持仓:")
            for pos in account_info['positions']:
                print(f"    {pos['coin']}: {pos['side']} | "
                      f"盈亏: ${pos['unrealized_pnl']:+,.2f}")

        # 3. AI分析市场
        print("\n[步骤 3/4] AI分析市场...")
        signal_datas = analyze_with_deepseek(all_coins_data, account_info)
        global signal_history
        signal_history.append(signal_datas)
        if len(signal_history) > 10:
            signal_history = signal_history[1:]

        # 4. 执行交易
        print("\n[步骤 4/4] 执行交易决策...")
        for signal_data in signal_datas:
            execute_trade(signal_data)

        print("\n" + "=" * 80)
        print("✓ 本次决策周期完成")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 交易机器人执行出错: {e}")
        import traceback
        traceback.print_exc()


def calculate_contract_amount(coin, margin_usd, leverage):
    """
    基于保证金和杠杆计算合约张数
    
    重要说明：本系统以USD为交易单位
    
    OKX永续合约说明：
    - 交易单位：USD（美元）
    - 每张合约代表一定数量的币种（contractSize）
    - 例如：BTC永续合约，每张 = 0.01 BTC
    - 计算逻辑：
      1. 名义价值(USD) = 保证金(USD) × 杠杆倍数
      2. 币种数量 = 名义价值 / 当前价格
      3. 合约张数 = 币种数量 / 每张合约对应的币数
    
    参数：
        coin: 币种名称（如'BTC', 'ETH'）
        margin_usd: 保证金金额，单位为USD（美元）
        leverage: 杠杆倍数（如1, 5, 10等）
    
    返回：
        包含合约张数、实际USD价值等信息的字典
    """
    symbol = get_symbol_for_coin(coin)

    # 计算名义价值 = 保证金 × 杠杆
    principal_usd = margin_usd * leverage

    # 获取市场信息
    market = exchange.market(symbol)

    # 获取交易限制
    min_amount = float(market['limits']['amount']['min'])  # 最小张数
    amount_precision = float(market['precision']['amount'])  # 张数精度
    contract_size = float(market.get('contractSize', 1))  # 每张合约代表的币数量

    # 获取当前价格
    ticker = exchange.fetch_ticker(symbol)
    current_price = float(ticker.get('last') or 0)

    if current_price <= 0:
        raise ValueError(f"无法获取 {coin} 的有效价格")

    # 计算最小USD价值要求
    # 最小USD = 最小张数 * 每张合约币数 * 当前价格
    min_usd_required = min_amount * contract_size * current_price

    print(f"市场信息:")
    print(f"  保证金(USD): ${margin_usd:.2f}")
    print(f"  杠杆倍数: {leverage}x")
    print(f"  名义价值(USD): ${principal_usd:.2f} (保证金 × 杠杆)")
    print(f"  最小张数: {min_amount}")
    print(f"  张数精度: {amount_precision}")
    print(f"  合约面值: {contract_size} {coin}")
    print(f"  当前价格: ${current_price:,.4f}")
    print(f"  最小USD要求: ${min_usd_required:,.2f}")

    if principal_usd < min_usd_required:
        min_margin_required = min_usd_required / leverage
        raise ValueError(
            f"名义价值 ${principal_usd:.2f} (保证金 ${margin_usd:.2f} × 杠杆 {leverage}x) 低于最小要求 ${min_usd_required:.2f}，"
            f"请提高保证金至少到 ${min_margin_required:.2f}"
        )

    # 计算合约张数（本系统以USD为单位，而非传统的币种数量）：
    # 步骤1：USD价值 -> 币种数量
    target_coin_amount = principal_usd / current_price

    # 步骤2：币种数量 -> 合约张数
    target_contracts = target_coin_amount / contract_size

    # 步骤3：使用交易所精度处理（向下取整到精度）
    try:
        adjusted_contracts = float(exchange.amount_to_precision(symbol, target_contracts))
    except Exception as e:
        # 如果 amount_to_precision 失败，手动处理精度
        print(f"⚠️ amount_to_precision 失败: {e}，使用手动精度处理")
        # 计算精度的倒数（例如精度0.01 -> 100）
        precision_multiplier = int(1 / amount_precision)
        adjusted_contracts = math.floor(target_contracts * precision_multiplier) / precision_multiplier

    # 确保不低于最小张数
    if adjusted_contracts < min_amount:
        print(f"⚠️ 计算张数 {adjusted_contracts} 低于最小要求 {min_amount}，使用最小张数")
        adjusted_contracts = min_amount

    # 计算实际USD价值
    actual_coin_amount = adjusted_contracts * contract_size
    actual_usd_value = actual_coin_amount * current_price  # 实际名义价值
    actual_margin = actual_usd_value / leverage  # 实际保证金

    print(f"\n计算结果:")
    print(f"  目标币数量: {target_coin_amount:.8f} {coin}")
    print(f"  目标张数: {target_contracts:.4f}")
    print(f"  调整后张数: {adjusted_contracts}")
    print(f"  实际币数量: {actual_coin_amount:.8f} {coin}")
    print(f"  实际名义价值(USD): ${actual_usd_value:,.2f}")
    print(f"  实际保证金(USD): ${actual_margin:,.2f}")

    return {
        'amount': adjusted_contracts,  # 合约张数
        'current_price': current_price,  # 当前价格(USD)
        'contract_size': contract_size,  # 每张合约对应的币数量
        'actual_usd_value': actual_usd_value,  # 实际名义价值(USD)
        'actual_margin': actual_margin,  # 实际保证金(USD)
        'actual_coin_amount': actual_coin_amount,  # 实际币数量
        'leverage': leverage,  # 杠杆倍数
    }


def open_long_with_margin(coin, margin_usd, leverage=1, take_profit_price: float = 0, stop_loss_price: float = 0):
    """
    根据保证金金额开多单（推荐使用此方法）
    
    重要说明：本系统以USD为交易单位，非传统的币种数量
    
    参数:
        coin: 币种，如 'BTC', 'ETH'
        margin_usd: 保证金金额，单位为USD（美元）
        leverage: 杠杆倍数
        take_profit_price: 止盈价格（可选）
        stop_loss_price: 止损价格（可选）
    
    说明：
        - 名义价值(USD) = 保证金(USD) × 杠杆倍数
        - 例如：保证金150 USD，杠杆5x → 名义价值 = 150×5 = 750 USD
    """
    principal_usd = margin_usd * leverage
    print(f"💡 保证金(USD): ${margin_usd:.2f}, 杠杆: {leverage}x → 名义价值(USD): ${principal_usd:.2f}")
    return _execute_open_position(coin, principal_usd, 'long', leverage, take_profit_price, stop_loss_price)


def open_short_with_margin(coin, margin_usd, leverage=1, take_profit_price: float = 0, stop_loss_price: float = 0):
    """
    根据保证金金额开空单（推荐使用此方法）
    
    重要说明：本系统以USD为交易单位，非传统的币种数量
    
    参数:
        coin: 币种，如 'BTC', 'ETH'
        margin_usd: 保证金金额，单位为USD（美元）
        leverage: 杠杆倍数
        take_profit_price: 止盈价格（可选）
        stop_loss_price: 止损价格（可选）
    
    说明：
        - 名义价值(USD) = 保证金(USD) × 杠杆倍数
        - 例如：保证金150 USD，杠杆5x → 名义价值 = 150×5 = 750 USD
    """
    principal_usd = margin_usd * leverage
    print(f"💡 保证金(USD): ${margin_usd:.2f}, 杠杆: {leverage}x → 名义价值(USD): ${principal_usd:.2f}")
    return _execute_open_position(coin, principal_usd, 'short', leverage, take_profit_price, stop_loss_price)


def _execute_open_position(coin, principal_usd, position_side, leverage, take_profit_price: float = 0,
                           stop_loss_price: float = 0):
    """
    执行开仓（内部函数）
    
    重要说明：本系统以USD为交易单位
    
    参数：
        coin: 币种
        principal_usd: 名义价值，单位为USD（美元）
        position_side: 仓位方向 ('long'或'short')
        leverage: 杠杆倍数
        take_profit_price: 止盈价格（可选）
        stop_loss_price: 止损价格（可选）
    """
    symbol = get_symbol_for_coin(coin)
    td_mode = TRADE_CONFIG.get('margin_mode', 'cross')
    side = 'buy' if position_side == 'long' else 'sell'

    print("\n" + "=" * 60)
    print(f"开仓 - {position_side.upper()}")
    print("=" * 60)
    print(f"币种: {coin}")
    print(f"交易对: {symbol}")
    print(f"方向: {position_side}")
    print(f"保证金模式: {td_mode}")
    print(f"杠杆: {leverage}x")
    if take_profit_price > 0:
        print(f"止盈价格: ${take_profit_price:,.4f}")
    if stop_loss_price > 0:
        print(f"止损价格: ${stop_loss_price:,.4f}")

    try:
        # 计算保证金（从名义价值反算）
        margin_usd = principal_usd / leverage

        # 计算合约数量（基于保证金和杠杆）
        calc_result = calculate_contract_amount(coin, margin_usd, leverage)
        amount = calc_result['amount']  # 合约张数
        current_price = calc_result['current_price']
        actual_notional = calc_result['actual_usd_value']  # 实际名义价值(USD)
        actual_margin = calc_result['actual_margin']  # 实际保证金(USD)

        print(f"\n最终下单参数:")
        print(f"  合约张数: {amount}")
        print(f"  实际名义价值(USD): ${actual_notional:,.2f}")
        print(f"  实际保证金(USD): ${actual_margin:,.2f}")

        # 设置杠杆
        try:
            exchange.set_leverage(
                leverage,
                symbol,
                {
                    'mgnMode': td_mode,
                    'posSide': position_side
                }
            )
            print(f"✓ 杠杆设置成功: {leverage}x")
        except Exception as leverage_error:
            print(f"⚠️ 设置杠杆失败: {leverage_error}")

        # 执行开仓
        params = {
            'tdMode': td_mode,
            'posSide': position_side,
        }
        
        order = exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=amount,
            params=params
        )

        print(f"✓ 开仓成功!")
        print(f"订单ID: {order.get('id')}")
        
        # 如果设置了止盈止损，使用OKX order-algo接口绑定
        # if take_profit_price > 0 or stop_loss_price > 0:
        #     try:
        #         market = exchange.market(symbol)
        #         inst_id = market['id']
        #         closing_side = 'sell' if position_side == 'long' else 'buy'
        #         ord_type = 'oco' if (take_profit_price > 0 and stop_loss_price > 0) else 'conditional'
        #         algo_request = {
        #             'instId': inst_id,
        #             'tdMode': td_mode,
        #             'side': closing_side,
        #             'posSide': position_side,
        #             'ordType': ord_type,
        #             'sz': exchange.amount_to_precision(symbol, amount),
        #             'reduceOnly': 'true',
        #         }
        #         if take_profit_price > 0:
        #             algo_request['tpTriggerPx'] = exchange.price_to_precision(symbol, take_profit_price)
        #             algo_request['tpTriggerPxType'] = 'last'
        #             algo_request['tpOrdPx'] = '-1'
        #         if stop_loss_price > 0:
        #             algo_request['slTriggerPx'] = exchange.price_to_precision(symbol, stop_loss_price)
        #             algo_request['slTriggerPxType'] = 'last'
        #             algo_request['slOrdPx'] = '-1'

        #         exchange.privatePostTradeOrderAlgo(algo_request)
        #         print(f"✓ 止盈止损设置已提交 (instId: {inst_id})")
        #         if take_profit_price > 0:
        #             print(f"  止盈触发价: ${take_profit_price:,.4f}")
        #         if stop_loss_price > 0:
        #             print(f"  止损触发价: ${stop_loss_price:,.4f}")
        #     except Exception as tpsl_error:
        #         print(f"⚠️ 止盈止损设置失败: {tpsl_error}")
        
        # 清空该币种的历史最大盈利记录（开新仓时重新开始记录）
        global max_profit_history
        if coin in max_profit_history:
            del max_profit_history[coin]
        max_profit_history[coin] = 0  # 初始化为0
        print(f"✓ 已重置 {coin} 的历史最大盈利记录")

        print("=" * 60)

        return order

    except Exception as e:
        print(f"❌ 开仓失败: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return None


def debug_positions():
    """调试函数：打印所有持仓的原始信息"""
    print("\n" + "=" * 60)
    print("调试持仓信息")
    print("=" * 60)
    try:
        positions = exchange.fetch_positions()
        print(f"总持仓数量: {len(positions)}")

        for i, pos in enumerate(positions):
            contracts = float(pos.get('contracts', 0) or 0)
            if contracts == 0 and pos.get('info'):
                contracts = float(pos['info'].get('pos', 0) or 0)
            lastUpdateTimestamp = float(pos.get('lastUpdateTimestamp')) / 1000
            # 毫秒时间戳转时间
            minute = math.ceil((datetime.now().timestamp() - lastUpdateTimestamp) / 60)
            print(minute)
            if contracts > 0:
                print(f"\n持仓 #{i + 1}:")
                print(f"  交易对: {pos.get('symbol')}")
                print(f"  数量(contracts): {pos.get('contracts')}")
                print(f"  方向(side): {pos.get('side')}")
                print(f"  info.pos: {pos.get('info', {}).get('pos')}")
                print(f"  info.posSide: {pos.get('info', {}).get('posSide')}")
                print(f"  info.mgnMode: {pos.get('info', {}).get('mgnMode')}")
                print(f"  原始info: {pos.get('info')}")

        print("=" * 60)
    except Exception as e:
        print(f"获取持仓信息失败: {e}")
        import traceback
        traceback.print_exc()


def close_position(coin, signal_data):
    """平仓（自动识别持仓方向）"""
    symbol = get_symbol_for_coin(coin)

    print("\n" + "=" * 60)
    print("平仓")
    print("=" * 60)
    print(f"币种: {coin}")
    print(f"交易对: {symbol}")

    try:
        # 获取当前持仓
        current_position = get_current_position(symbol)

        if not current_position:
            print(f"⚠️ {coin} 无持仓，无法平仓")
            print("=" * 60)
            return None

        position_side = current_position['side']
        close_amount = float(current_position['quantity'])
        td_mode = current_position.get('margin_mode', TRADE_CONFIG.get('margin_mode', 'cross'))

        print(f"持仓方向: {position_side}")
        print(f"持仓数量(合约张数): {close_amount}")
        print(f"保证金模式: {td_mode}")
        print(f"入场价格(USD): ${current_position['entry_price']:,.2f}")
        print(f"当前价格(USD): ${current_position['current_price']:,.2f}")
        print(f"未实现盈亏(USD): ${current_position['unrealized_pnl']:+,.2f}")

        # 执行平仓（反向下单）
        # 注意：OKX 对冲模式下，平多仓用 sell，平空仓用 buy
        side = 'sell' if position_side == 'long' else 'buy'

        print(f"\n准备平仓:")
        print(f"  下单方向(side): {side}")
        print(f"  持仓方向(posSide): {position_side}")
        print(f"  平仓数量: {close_amount}")

        order = exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=close_amount,
            params={
                'reduceOnly': True,
                'tdMode': td_mode,
                'posSide': position_side,
            }
        )

        print(f"\n✓ 平仓成功!")
        print(f"订单ID: {order.get('id')}")
        print("=" * 60)
        # 清理币种数据
        global coin_data
        coin_data[coin] = {}
        # 清理历史最大盈利记录
        global max_profit_history
        if coin in max_profit_history:
            del max_profit_history[coin]
            print(f"✓ 已清除 {coin} 的历史最大盈利记录")
        global sharpe_ratio
        sharpe_ratio = calculate_sharpe_ratio(50)
        return order

    except Exception as e:
        print(f"\n❌ 平仓失败: {e}")
        print(f"错误详情:")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return None


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(f"{'多币种AI量化交易系统启动':^80}")
    print("=" * 80)
    print(f"\n配置信息:")
    print(f"  支持币种: {', '.join(TRADE_CONFIG['coins'])}")
    print(f"  决策频率: {TRADE_CONFIG['decision_interval']}秒 ({TRADE_CONFIG['decision_interval'] // 60}分钟)")
    print(f"  日内数据: {TRADE_CONFIG['timeframes']['intraday']}")
    print(f"  背景数据: {TRADE_CONFIG['timeframes']['background']}")
    print(f"  杠杆范围: {TRADE_CONFIG['leverage_range']}x")
    print(f"  起始资金: ${TRADE_CONFIG['initial_capital']:,.2f}")
    print(f"  测试模式: {'是' if TRADE_CONFIG['test_mode'] else '否'}")
    print(f"  交易所K线源: {exchange.timeframes}")
    print(f"\n系统启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 循环执行
    global current_step
    global sharpe_ratio
    sharpe_ratio = calculate_sharpe_ratio(50)
    while True:
        try:
            # 执行交易决策
            trading_bot()
            current_step += 1

            # 等待到下一个决策周期
            wait_seconds = wait_for_next_period(TRADE_CONFIG['decision_interval'] // 60)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
        except KeyboardInterrupt:
            print("\n\n用户中断，正在退出...")
            break
        except Exception as e:
            print(f"\n主循环出错: {e}")
            import traceback
            traceback.print_exc()
            print("等待60秒后重试...")
            time.sleep(60)


if __name__ == "__main__":
    main()
    # open_short_with_margin('BTC', 50, 1, 90000, 200000)
