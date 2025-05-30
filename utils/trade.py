import numpy as np
from datetime import datetime


def buy_sell_smart(today, pred, balance, shares, risk=5):
    """
    Smart trading logic that now supports both long and short positions.

    Args:
        today (float): Current price.
        pred (float): Predicted price.
        balance (float): Current cash balance.
        shares (float): Number of shares held (can be negative for short positions).
        risk (float): Risk factor (%).

    Returns:
        tuple: (new_balance, new_shares)
    """
    diff = pred * risk / 100
    # Long/Buy logic
    if today > pred + diff:
        if shares < 0:  # Close out short first
            balance += abs(shares) * today  # Cover the short
            shares = 0
        balance += shares * today  # Sell off long position
        shares = 0

    elif today > pred:
        if shares < 0:  # Close out short first
            balance += (
                abs(shares) * today * (today - pred) / diff
            )  # Cover partial short
            shares *= 1 - (today - pred) / diff

        if shares > 0:  # Sell partial long position
            factor = (today - pred) / diff
            balance += shares * factor * today
            shares *= 1 - factor

    # Short/Sell Logic
    elif today < pred - diff:
        if shares > 0:  # Close out long first
            balance += shares * today
            shares = 0

        shares += balance / today  # Open short position
        balance = 0

    elif today < pred:
        if shares > 0:  # Close out long first
            balance += shares * today * (pred - today) / diff
            shares *= 1 - (pred - today) / diff

        if shares < 0:  # Increase short position
            factor = (pred - today) / diff
            shares += balance * factor / today
            balance *= 1 - factor

    return balance, shares


def buy_sell_smart_w_short(today, pred, balance, shares, risk=5, max_n_btc=0.002):
    """
    Smart trading with explicit short position management using a max_n_btc limit.
    Args:
        today (float): Current price.
        pred (float): Predicted price.
        balance (float): Current cash balance.
        shares (float): Number of shares held (can be negative for short positions).
        risk (float): Risk factor (%).
        max_n_btc (float): Maximum number of BTC to short.

    Returns:
        tuple: (new_balance, new_shares)
    """
    diff = pred * risk / 100

    # Short Entry and Increase logic
    if today < pred - diff:
        if shares > 0:
            balance += shares * today
            shares = 0

        # Calculate the value of short position to be initiated
        short_amount_value = balance

        # Calculate the number of shares to short sell based on available balance
        short_shares = short_amount_value / today

        if shares >= 0:
            # Initialize short position
            shares += short_shares
            balance = 0

    elif today < pred:
        if shares > 0:  # Close out long first
            balance += shares * today * (pred - today) / diff
            shares *= 1 - (pred - today) / diff

        if shares < 0:  # Increase short position
            factor = (pred - today) / diff
            shares += balance * factor / today
            balance *= 1 - factor

    # Long Entry and Increase Logic
    elif today > pred + diff:
        if shares < 0:  # Close out short first
            balance += abs(shares) * today
            shares = 0

        shares += balance / today
        balance = 0

    elif today > pred:
        if shares < 0:  # Close out short first
            balance += abs(shares) * today * (today - pred) / diff
            shares *= 1 - (today - pred) / diff

        if shares > 0:  # Sell partial long position
            factor = (today - pred) / diff
            balance += shares * factor * today
            shares *= 1 - factor

    return balance, shares


def buy_sell_vanilla(today, pred, balance, shares, tr=0.01):
    """
    Vanilla trading logic that now supports both long and short positions.

    Args:
        today (float): Current price.
        pred (float): Predicted price.
        balance (float): Current cash balance.
        shares (float): Number of shares held (can be negative for short positions).
        tr (float): Threshold for price change (%).

    Returns:
        tuple: (new_balance, new_shares)
    """
    price_diff_ratio = abs((pred - today) / today)

    if price_diff_ratio < tr:
        return balance, shares  # Hold if change is below threshold

    # Long logic
    if pred > today:  # Predicted price increase -> Buy
        if shares < 0:
            balance += abs(shares) * today  # Cover short position
            shares = 0
        if balance > 0:  # Only buy if there is cash
            shares += balance / today  # Buy long
            balance = 0

    # Short logic
    elif pred < today:  # Predicted price decrease -> Sell
        if shares > 0:
            balance += shares * today  # Sell long position
            shares = 0
        if balance > 0:  # Only short if there is cash to back it (conceptually)
            shares -= balance / today  # Sell short (shares become negative)
            balance = 0  # Balance used to initiate short

    return balance, shares


# --- New Strategy Function ---
def buy_sell_timed_exit(
    today_price,
    pred_price,
    balance,
    shares,  # Can be positive (long), negative (short), or zero
    position_active,  # True if a position (long or short) is open
    position_type,  # 'long', 'short', or None
    entry_price,
    entry_timestamp,
    current_timestamp,
    time_resolution_seconds,
    max_hold_period,  # In multiples of time resolution
    tp,  # Take profit % (e.g., 0.05 for 5%)
    sl,  # Stop loss % (e.g., 0.02 for 2%)
    tc,  # Transaction cost % (e.g., 0.001 for 0.1%)
    signal_threshold_long=0.0,  # Minimum predicted relative increase for long entry
    signal_threshold_short=0.0,  # Minimum predicted relative decrease for short entry
):
    """
    Implements a trading strategy with timed exit, TP, SL for both long and short.

    Args:
        today_price (float): Current market price for decision making.
        pred_price (float): Predicted price for the next period (used for entry signal).
        balance (float): Current cash balance.
        shares (float): Current number of shares held (positive=long, negative=short).
        position_active (bool): Whether a position is currently open.
        position_type (str | None): 'long', 'short', or None.
        entry_price (float): Price at which the current position was entered.
        entry_timestamp (int): Timestamp when the current position was entered.
        current_timestamp (int): Current timestamp of the data point.
        time_resolution_seconds (int): Time difference between data points in seconds.
        max_hold_period (int): Maximum holding period in multiples of time resolution.
        tp (float): Take profit percentage (positive value).
        sl (float): Stop loss percentage (positive value).
        tc (float): Transaction cost percentage (positive value).
        signal_threshold_long (float): Min predicted relative increase (pred/today - 1) for long.
        signal_threshold_short (float): Min predicted relative decrease (1 - pred/today) for short.

    Returns:
        tuple: (new_balance, new_shares, new_position_active, new_position_type, new_entry_price, new_entry_timestamp)
    """
    new_balance = balance
    new_shares = shares
    new_position_active = position_active
    new_position_type = position_type
    new_entry_price = entry_price
    new_entry_timestamp = entry_timestamp

    # Ensure percentages are positive
    tp = abs(tp)
    sl = abs(sl)
    tc = abs(tc)

    max_hold_duration_seconds = max_hold_period * time_resolution_seconds

    if position_active:
        # --- Check Exit Conditions ---
        holding_duration = current_timestamp - entry_timestamp
        liquidate = False
        reason = ""

        # 1. Max Hold Period
        if holding_duration >= max_hold_duration_seconds:
            liquidate = True
            reason = f"Max Hold ({holding_duration/60:.1f}m / {max_hold_duration_seconds/60:.1f}m)"

        # 2. Take Profit & Stop Loss (depends on position type)
        elif position_type == "long":
            if today_price >= entry_price * (1 + tp):
                liquidate = True
                reason = f"Long TP ({tp*100:.2f}%)"
            elif today_price <= entry_price * (1 - sl):
                liquidate = True
                reason = f"Long SL ({sl*100:.2f}%)"
        elif position_type == "short":
            if today_price <= entry_price * (1 - tp):  # Price decreased for profit
                liquidate = True
                reason = f"Short TP ({tp*100:.2f}%)"
            elif today_price >= entry_price * (1 + sl):  # Price increased for loss
                liquidate = True
                reason = f"Short SL ({sl*100:.2f}%)"

        if liquidate:
            # print(f"[{datetime.fromtimestamp(current_timestamp)}] EXIT {position_type.upper()} {reason}. Price: {today_price:.2f}, Entry: {entry_price:.2f}") # Debug
            if position_type == "long":
                # Sell long shares, deduct transaction cost from proceeds
                proceeds = new_shares * today_price
                new_balance += proceeds * (1 - tc)
            elif position_type == "short":
                # Buy back short shares, add transaction cost to cost
                cost_to_cover = abs(new_shares) * today_price
                new_balance -= cost_to_cover * (1 + tc)
            # Reset position state
            new_shares = 0
            new_position_active = False
            new_position_type = None
            new_entry_price = 0
            new_entry_timestamp = 0
        # else: # Debug
        # print(f"[{datetime.fromtimestamp(current_timestamp)}] HOLD {position_type.upper()}. Price: {today_price:.2f}, Entry: {entry_price:.2f}, Hold: {holding_duration/60:.1f}m")

    else:  # No position active, check for entry signals
        # --- Check Entry Conditions ---
        predicted_increase_ratio = (pred_price / today_price) - 1
        predicted_decrease_ratio = 1 - (pred_price / today_price)

        enter_long = predicted_increase_ratio > signal_threshold_long
        enter_short = predicted_decrease_ratio > signal_threshold_short

        # Prioritize one if both signals occur (optional, here we prioritize long)
        if enter_long and enter_short:
            # Example: prioritize based on magnitude, or simply choose one
            if predicted_increase_ratio >= predicted_decrease_ratio:
                enter_short = False
            else:
                enter_long = False

        if new_balance > 1e-6:  # Only enter if we have a non-negligible balance
            if enter_long:
                # print(f"[{datetime.fromtimestamp(current_timestamp)}] ENTER LONG. Pred: {pred_price:.2f}, Today: {today_price:.2f}") # Debug
                buy_amount_gross = new_balance  # Use full balance
                # Calculate shares bought after deducting transaction cost
                new_shares = (buy_amount_gross / today_price) * (1 - tc)
                new_balance = 0
                new_position_active = True
                new_position_type = "long"
                new_entry_price = today_price
                new_entry_timestamp = current_timestamp
            elif enter_short:
                # print(f"[{datetime.fromtimestamp(current_timestamp)}] ENTER SHORT. Pred: {pred_price:.2f}, Today: {today_price:.2f}") # Debug
                short_amount_gross = (
                    new_balance  # Use full balance as basis for short value
                )
                # Calculate shares sold short after deducting transaction cost conceptually
                # The balance increases by the proceeds of the short sale (minus cost)
                # We represent the short position with negative shares
                proceeds = short_amount_gross * (
                    1 - tc
                )  # Effective value received after cost
                new_shares = -(
                    proceeds / today_price
                )  # Negative shares corresponding to proceeds
                # Balance increases by the value shorted (minus cost), shares become negative
                new_balance = proceeds  # Balance now reflects the cash from shorting
                new_position_active = True
                new_position_type = "short"
                new_entry_price = today_price
                new_entry_timestamp = current_timestamp
        # else: # Debug
        # if enter_long or enter_short:
        # print(f"[{datetime.fromtimestamp(current_timestamp)}] Signal but no balance. Pred: {pred_price:.2f}, Today: {today_price:.2f}")

    return (
        new_balance,
        new_shares,
        new_position_active,
        new_position_type,
        new_entry_price,
        new_entry_timestamp,
    )


def buy_sell_timed_exit_futures(
    today_price,
    pred_price,
    margin_balance,  # 변경: balance -> margin_balance
    position_size,  # 변경: shares -> position_size (양수: long, 음수: short)
    position_active,
    position_type,  # 'long', 'short', or None
    entry_price,
    entry_timestamp,
    current_timestamp,
    time_resolution_seconds,
    max_hold_period,
    tp,  # Take profit %
    sl,  # Stop loss %
    tc,  # Transaction cost % (명목 가치 기준)
    leverage,  # 추가: 레버리지
    maintenance_margin_rate,  # 추가: 유지 증거금 비율 (e.g., 0.005 for 0.5%)
    signal_threshold_long,
    signal_threshold_short,
    risk_per_trade=1.0,  # 추가: 진입 시 사용할 증거금 비율 (e.g., 1.0 = 모든 증거금 사용)
):
    """
    선물 거래 전략: Timed Exit, TP, SL, Liquidation, Leverage 적용

    Args:
        ... (기존 Args 설명 참조, balance/shares 변경됨) ...
        leverage (float): 사용할 레버리지 배율.
        maintenance_margin_rate (float): 포지션 유지를 위한 최소 증거금 비율.
        risk_per_trade (float): 포지션 진입 시 사용할 가용 증거금의 비율 (0.0 ~ 1.0).

    Returns:
        tuple: (new_margin_balance, new_position_size, new_position_active, new_position_type, new_entry_price, new_entry_timestamp)
    """
    new_margin_balance = margin_balance
    new_position_size = position_size
    new_position_active = position_active
    new_position_type = position_type
    new_entry_price = entry_price
    new_entry_timestamp = entry_timestamp

    tp = abs(tp)
    sl = abs(sl)
    tc = abs(tc)

    max_hold_duration_seconds = max_hold_period * time_resolution_seconds

    liquidated = False
    liquidation_price = None  # 정보용

    if position_active:
        # --- 청산 조건 확인 (가장 먼저) ---
        current_notional_value = abs(position_size) * today_price
        if current_notional_value > 1e-9:  # 포지션 가치가 0이 아닐 때만 계산
            unrealized_pnl = (
                today_price - entry_price
            ) * position_size  # 숏 포지션은 음수 PnL이 이익
            equity = margin_balance + unrealized_pnl
            required_maintenance_margin = (
                current_notional_value * maintenance_margin_rate
            )

            if equity < required_maintenance_margin:
                liquidated = True
                # 실제 청산 가격은 더 불리할 수 있으나, 시뮬레이션에서는 현재가 사용
                liquidation_price = today_price
                # 청산 시 증거금은 0으로 처리 (단순화)
                # print(f"[{datetime.fromtimestamp(current_timestamp)}] *** LIQUIDATION *** Type: {position_type}, Price: {today_price:.2f}, Entry: {entry_price:.2f}, Equity: {equity:.2f}, Required MM: {required_maintenance_margin:.2f}") # Debug
                new_margin_balance = 0  # 증거금 손실
                new_position_size = 0
                new_position_active = False
                new_position_type = None
                new_entry_price = 0
                new_entry_timestamp = 0

        # --- 청산되지 않았다면, 종료 조건 확인 ---
        if not liquidated:
            holding_duration = current_timestamp - entry_timestamp
            close_position = False
            reason = ""

            # 1. Max Hold Period
            if holding_duration >= max_hold_duration_seconds:
                close_position = True
                reason = f"Max Hold ({holding_duration/60:.1f}m / {max_hold_duration_seconds/60:.1f}m)"

            # 2. Take Profit & Stop Loss
            elif position_type == "long":
                if today_price >= entry_price * (1 + tp):
                    close_position = True
                    reason = f"Long TP ({tp*100:.2f}%)"
                elif today_price <= entry_price * (1 - sl):
                    close_position = True
                    reason = f"Long SL ({sl*100:.2f}%)"
            elif position_type == "short":
                if today_price <= entry_price * (1 - tp):  # 가격 하락 시 이익
                    close_position = True
                    reason = f"Short TP ({tp*100:.2f}%)"
                elif today_price >= entry_price * (1 + sl):  # 가격 상승 시 손실
                    close_position = True
                    reason = f"Short SL ({sl*100:.2f}%)"

            if close_position:
                # print(f"[{datetime.fromtimestamp(current_timestamp)}] EXIT {position_type.upper()} {reason}. Price: {today_price:.2f}, Entry: {entry_price:.2f}") # Debug
                # 포지션 종료: 실현 손익 계산 및 수수료 차감
                realized_pnl = (today_price - entry_price) * position_size
                exit_notional_value = abs(position_size) * today_price
                transaction_cost_exit = exit_notional_value * tc

                new_margin_balance += (
                    realized_pnl - transaction_cost_exit
                )  # 증거금 업데이트
                # Reset position state
                new_position_size = 0
                new_position_active = False
                new_position_type = None
                new_entry_price = 0
                new_entry_timestamp = 0
            # else: # Debug
            #     unrealized_pnl = (today_price - entry_price) * position_size
            #     equity = margin_balance + unrealized_pnl
            #     print(f"[{datetime.fromtimestamp(current_timestamp)}] HOLD {position_type.upper()}. Price: {today_price:.2f}, Entry: {entry_price:.2f}, Hold: {holding_duration/60:.1f}m, Equity: {equity:.2f}")

    # --- 포지션이 없고 청산되지 않았다면, 진입 조건 확인 ---
    if (
        not position_active and not liquidated and margin_balance > 1e-9
    ):  # 증거금이 있을 때만 진입 시도
        predicted_increase_ratio = (pred_price / today_price) - 1
        predicted_decrease_ratio = 1 - (pred_price / today_price)

        enter_long = predicted_increase_ratio > signal_threshold_long
        enter_short = predicted_decrease_ratio > signal_threshold_short

        # 우선순위 결정 (여기서는 간단히 롱 우선)
        if enter_long and enter_short:
            if predicted_increase_ratio >= predicted_decrease_ratio:
                enter_short = False
            else:
                enter_long = False

        if enter_long or enter_short:
            # 진입 로직: 레버리지 적용, 수수료 계산
            margin_to_use = margin_balance * risk_per_trade
            notional_value_to_open = margin_to_use * leverage
            position_size_to_open = notional_value_to_open / today_price
            transaction_cost_entry = notional_value_to_open * tc

            # 필요한 초기 증거금 = 명목가치 / 레버리지. 사용할 증거금(margin_to_use)보다 작거나 같아야 함.
            # 또한, 진입 수수료를 지불할 증거금이 남아 있어야 함.
            # 여기서는 margin_to_use가 이미 가용 증거금의 일부이므로, 수수료만 추가 확인.
            if (
                margin_balance >= transaction_cost_entry
            ):  # 실제로는 margin_to_use + tc 보다 커야 하지만, risk_per_trade로 조절했다고 가정
                # print(f"[{datetime.fromtimestamp(current_timestamp)}] ENTER {'LONG' if enter_long else 'SHORT'}. Pred: {pred_price:.2f}, Today: {today_price:.2f}, Size: {position_size_to_open:.4f}") # Debug

                new_margin_balance -= transaction_cost_entry  # 진입 수수료 차감
                new_position_size = (
                    position_size_to_open if enter_long else -position_size_to_open
                )
                new_position_active = True
                new_position_type = "long" if enter_long else "short"
                new_entry_price = today_price
                new_entry_timestamp = current_timestamp
            # else: # Debug
            # print(f"[{datetime.fromtimestamp(current_timestamp)}] Signal but insufficient margin for entry cost. Need: {transaction_cost_entry:.4f}, Have: {margin_balance:.4f}")

    # 청산된 경우, 이미 new_* 변수들이 업데이트 되었으므로 그대로 반환
    return (
        new_margin_balance,
        new_position_size,
        new_position_active,
        new_position_type,
        new_entry_price,
        new_entry_timestamp,
    )


def trade(
    data,
    time_key,
    timstamps,
    targets,
    preds,
    initial_margin=100,  # 변경: balance -> initial_margin
    mode="timed_exit",
    # risk 파라미터는 smart/vanilla 모드에서는 계속 사용될 수 있음
    risk=5,
    y_key="Close",
    # --- 선물 거래용 파라미터 ---
    leverage=1.0,  # 추가: 레버리지 배율
    maintenance_margin_rate=0.005,  # 추가: 유지 증거금 비율 (0.5%)
    # --- timed_exit_futures 모드용 파라미터 ---
    max_hold_period=30,
    tp=0.04,
    sl=0.01,
    tc=0.0004,  # 변경/추가: 선물 거래 수수료 (시장가 0.04% 가정)
    signal_threshold_long=0.01,
    signal_threshold_short=0.01,
    risk_per_trade=1.0,  # 추가: timed_exit_futures 진입 시 사용할 증거금 비율
    # --- smart_w_short 모드용 파라미터 ---
    max_n_btc=0.002,  # 이 파라미터는 선물 컨셉과 직접적 관련은 적어짐
):
    """
    선물 거래 시뮬레이션 (timed_exit 모드는 선물 로직 적용).

    Args:
        ... (기존 Args 설명 참조, balance 변경됨) ...
        initial_margin (float): 초기 증거금.
        leverage (float): 사용할 레버리지 배율 ('timed_exit' 모드).
        maintenance_margin_rate (float): 유지 증거금 비율 ('timed_exit' 모드).
        max_hold_period (int): 최대 보유 기간 ('timed_exit' 모드).
        tp (float): Take profit % ('timed_exit' 모드).
        sl (float): Stop loss % ('timed_exit' 모드).
        tc (float): 거래 수수료 % (명목 가치 기준, 'timed_exit' 모드).
        signal_threshold_long (float): Long 진입 시그널 임계값 ('timed_exit' 모드).
        signal_threshold_short (float): Short 진입 시그널 임계값 ('timed_exit' 모드).
        risk_per_trade (float): 진입 시 사용할 증거금 비율 ('timed_exit' 모드).
        max_n_btc (float): 'smart_w_short' 모드용 파라미터.

    Returns:
        tuple: (final_margin_balance, list_of_equity_over_time)
    """
    margin_balance = initial_margin  # 초기 증거금 설정
    # 포트폴리오 가치(Equity)를 추적 (증거금 + 미실현 손익)
    equity_in_time = [initial_margin]
    position_size = 0.0  # 포지션 크기 (양수: long, 음수: short)

    # State variables for timed_exit_futures mode
    position_active = False
    position_type = None
    entry_price = 0.0
    entry_timestamp = 0

    # --- 입력 데이터 정렬 및 시간 해상도 계산 (기존과 동일) ---
    if len(timstamps) != len(targets) or len(timstamps) != len(preds):
        raise ValueError(
            "Timestamps, targets, and preds lists must have the same length."
        )
    if len(timstamps) < 2:
        print("Warning: Less than 2 data points. Cannot run simulation.")
        return initial_margin, equity_in_time

    sorted_indices = np.argsort(timstamps)
    timestamps_sorted = np.array(timstamps)[sorted_indices]
    targets_sorted = np.array(targets)[sorted_indices]
    preds_sorted = np.array(preds)[sorted_indices]

    time_diffs = np.diff(timestamps_sorted)
    if not np.all(time_diffs > 0):
        unique_ts, unique_indices = np.unique(timestamps_sorted, return_index=True)
        if len(unique_ts) < 2:
            print("Error: Not enough unique timestamps after filtering.")
            return initial_margin, equity_in_time
        timestamps_sorted = unique_ts
        targets_sorted = targets_sorted[unique_indices]
        preds_sorted = preds_sorted[unique_indices]
        time_diffs = np.diff(timestamps_sorted)

    time_resolution_seconds = int(np.median(time_diffs))
    if time_resolution_seconds <= 0:
        if len(time_diffs) > 0 and np.any(time_diffs > 0):
            time_resolution_seconds = int(np.mean(time_diffs[time_diffs > 0]))
        elif len(timestamps_sorted) >= 2:
            time_resolution_seconds = int(timestamps_sorted[1] - timestamps_sorted[0])
        else:
            raise ValueError(
                f"Could not determine a positive time resolution. Timestamps: {timestamps_sorted}"
            )

    # --- 파라미터 유효성 검사 (timed_exit 모드) ---
    if mode == "timed_exit":
        if not all(
            p is not None
            for p in [max_hold_period, tp, sl, tc, leverage, maintenance_margin_rate]
        ):
            raise ValueError("Missing parameters for 'timed_exit' futures mode.")
        if (
            max_hold_period <= 0
            or tc < 0
            or tp <= 0
            or sl <= 0
            or leverage <= 0
            or maintenance_margin_rate <= 0
        ):
            raise ValueError(
                "Invalid parameters for 'timed_exit' futures mode (must be positive, tc non-negative)."
            )
        if (
            signal_threshold_long < 0
            or signal_threshold_short < 0
            or risk_per_trade <= 0
            or risk_per_trade > 1.0
        ):
            raise ValueError(
                "Invalid signal thresholds or risk_per_trade for 'timed_exit'."
            )

    # --- 시뮬레이션 루프 ---
    for i in range(len(timestamps_sorted)):
        current_ts = int(timestamps_sorted[i])
        # current_target: 현재 시점 *종료* 시 가격 (가치 평가용)
        current_target_price = targets_sorted[i]
        # current_pred: 현재 시점 *시작 전* 데이터 기반 예측 (다음 시점 예측)
        current_pred = preds_sorted[i]

        # --- 'today_price': 의사결정 시점(현재 시점 *시작*)의 가격 ---
        if i == 0:
            decision_ts = int(current_ts - time_resolution_seconds)
        else:
            decision_ts = int(timestamps_sorted[i - 1])

        today_data_row = data[data[time_key] == decision_ts]

        if today_data_row.empty:
            # 데이터 누락 시 처리 (기존과 유사하게 처리, 단 가치평가는 equity 기준)
            print(
                f"Warning: Missing price data for decision timestamp {decision_ts}. Holding position."
            )
            # 이전 equity 값을 그대로 사용
            if len(equity_in_time) > 0:
                equity_in_time.append(equity_in_time[-1])
            else:  # 첫 스텝부터 누락 시 초기 마진 사용
                equity_in_time.append(initial_margin)
            continue
        else:
            today_price = today_data_row.iloc[0][y_key]
            if (
                not isinstance(today_price, (int, float))
                or np.isnan(today_price)
                or today_price <= 0
            ):
                print(
                    f"Warning: Invalid 'today_price' ({today_price}) for {decision_ts}. Holding position."
                )
                if len(equity_in_time) > 0:
                    equity_in_time.append(equity_in_time[-1])
                else:
                    equity_in_time.append(initial_margin)
                continue

        # --- 거래 전략 실행 ---
        if mode == "smart":
            # 현물 로직: balance 대신 margin_balance, shares 대신 position_size 사용 (개념적 불일치 주의)
            margin_balance, position_size = buy_sell_smart(
                today_price, current_pred, margin_balance, position_size, risk=risk
            )
        elif mode == "smart_w_short":
            # 현물 로직: balance 대신 margin_balance, shares 대신 position_size 사용 (개념적 불일치 주의)
            margin_balance, position_size = buy_sell_smart_w_short(
                today_price,
                current_pred,
                margin_balance,
                position_size,
                risk=risk,
                max_n_btc=max_n_btc,
            )
        elif mode == "vanilla":
            # 현물 로직: balance 대신 margin_balance, shares 대신 position_size 사용 (개념적 불일치 주의)
            tr_threshold = risk / 100.0
            margin_balance, position_size = buy_sell_vanilla(
                today_price,
                current_pred,
                margin_balance,
                position_size,
                tr=tr_threshold,
            )
        elif mode == "no_strategy":
            # 현물 Buy & Hold 유사 로직 (개념적 불일치 주의)
            if i == 0 and margin_balance > 0:
                # 레버리지 없이 전액 매수하는 현물처럼 동작 (선물 컨셉 아님)
                position_size += margin_balance / today_price
                margin_balance = 0
        elif mode == "timed_exit":
            # 선물 거래 로직 실행
            (
                margin_balance,
                position_size,
                position_active,
                position_type,
                entry_price,
                entry_timestamp,
            ) = buy_sell_timed_exit_futures(
                today_price,
                current_pred,
                margin_balance,
                position_size,
                position_active,
                position_type,
                entry_price,
                entry_timestamp,
                current_ts,
                time_resolution_seconds,
                max_hold_period,
                tp,
                sl,
                tc,
                leverage,  # 전달
                maintenance_margin_rate,  # 전달
                signal_threshold_long,
                signal_threshold_short,
                risk_per_trade,  # 전달
            )
        else:
            raise ValueError(f"Unknown trade mode: {mode}")

        # --- 포트폴리오 가치 평가 (Equity 계산) ---
        # 현재 시점 *종료* 가격(current_target_price) 기준
        current_equity = margin_balance  # 기본은 증거금
        if position_active and abs(position_size) > 1e-9:
            unrealized_pnl = (current_target_price - entry_price) * position_size
            current_equity += unrealized_pnl  # 미실현 손익 반영

        # 청산 등으로 equity가 음수가 될 수 있으나, 최소 0으로 제한 (선택적)
        current_equity = max(0, current_equity)
        equity_in_time.append(current_equity)

        # 다음 스텝을 위해 상태 업데이트 (이미 strategy 함수 내에서 new_* 변수로 처리됨)

    # --- 최종 청산 ---
    # 시뮬레이션 종료 시점에 포지션이 남아있으면 시장가로 종료
    final_price = targets_sorted[-1]
    if position_active and abs(position_size) > 1e-9:
        # print(f"Final Liquidation ({position_type}) at {final_price:.2f}") # Debug
        realized_pnl = (final_price - entry_price) * position_size
        exit_notional_value = abs(position_size) * final_price
        # timed_exit 모드였을 경우에만 수수료 적용 (다른 모드는 수수료 개념 없음)
        final_tc = tc if mode == "timed_exit" else 0.0
        transaction_cost_exit = exit_notional_value * final_tc

        margin_balance += realized_pnl - transaction_cost_exit
        position_size = 0  # 포지션 종료

    # 최종 증거금 잔고 (음수 방지)
    final_margin_balance = max(0, margin_balance)

    # equity_in_time의 마지막 값을 최종 증거금으로 업데이트
    if len(equity_in_time) > len(
        timestamps_sorted
    ):  # 이미 마지막 스텝의 equity가 추가된 경우
        equity_in_time[-1] = final_margin_balance
    else:  # 마지막 스텝의 equity가 누락된 경우 (이론상 발생 안 함)
        equity_in_time.append(final_margin_balance)

    # smart/vanilla 모드는 balance/shares 개념이므로 반환값 주의 필요
    # 여기서는 timed_exit 기준에 맞춰 final_margin_balance 와 equity_in_time 반환
    return final_margin_balance, equity_in_time
