from game.simulator import PricingGame
from game.simulator import PricingGame
import numpy as np
import pulp


def is_in_optimization(current_cw, cw, start_dates, end_dates, include_future_articles):
    in_optimization_horizon = np.logical_and(start_dates <= cw, cw <= end_dates)
    currently_online = current_cw >= start_dates
    if not include_future_articles:
        in_optimization_horizon = np.logical_and(in_optimization_horizon, currently_online)
    return in_optimization_horizon


def add_profit_penalty_objective(model_dict, game):
    model = model_dict["model"]
    target_profit_ratio = game.target_profit_ratio
    penalty = game.profit_lack_penalty

    # profit and revenue so far
    revenue_so_far = sum([r.sum() for r in game.revenues])
    profit_so_far = sum([r.sum() for r in game.profits])

    profit_lack = pulp.LpVariable("profit_lack", lowBound=0.0, cat="Continuous")
    model += profit_lack >= target_profit_ratio * (
        model_dict["inseason_revenue"] + model_dict["residual_revenue"] + revenue_so_far
    ) - (model_dict["inseason_profit"] + model_dict["residual_profit"] + profit_so_far)

    # objective function
    model += model_dict["inseason_revenue"] + model_dict["residual_revenue"] + revenue_so_far - penalty * profit_lack


def add_profit_objective(model_dict, game):
    model = model_dict["model"]
    profit_so_far = sum([r.sum() for r in game.profits])

    # objective function
    model += model_dict["inseason_profit"] + model_dict["residual_profit"] + profit_so_far


def add_revenue_objective(model_dict, game):
    model = model_dict["model"]
    # profit and revenue so far
    revenue_so_far = sum([r.sum() for r in game.revenues])

    # objective function
    model += model_dict["inseason_revenue"] + model_dict["residual_revenue"] + revenue_so_far


def add_sdr_target_constraint(model_dict, game, sdr_lb, sdr_ub, obs, num_discounts, include_future_articles):
    num_products = game.num_products
    black_prices = obs["black_prices"]
    sales = model_dict["sales"]
    article_season_start = obs["article_season_start"]
    article_season_end = obs["article_season_end"]
    cw = obs["cw"]
    model = model_dict["model"]
    discount_range = np.linspace(0, 0.7, num_discounts)

    model += pulp.lpSum(
        [
            black_prices[i] * discount_range[j] * sales[i, cw, j]
            for i in range(num_products)
            for j in range(num_discounts)
            if is_in_optimization(cw, cw, article_season_start, article_season_end, include_future_articles)[i]
        ]
    ) >= sdr_lb * pulp.lpSum(
        [
            black_prices[i] * sales[i, cw, j]
            for i in range(num_products)
            for j in range(num_discounts)
            if is_in_optimization(cw, cw, article_season_start, article_season_end, include_future_articles)[i]
        ]
    )

    model += pulp.lpSum(
        [
            black_prices[i] * discount_range[j] * sales[i, cw, j]
            for i in range(num_products)
            for j in range(num_discounts)
            if is_in_optimization(cw, cw, article_season_start, article_season_end, include_future_articles)[i]
        ]
    ) <= sdr_ub * pulp.lpSum(
        [
            black_prices[i] * sales[i, cw, j]
            for i in range(num_products)
            for j in range(num_discounts)
            if is_in_optimization(cw, cw, article_season_start, article_season_end, include_future_articles)[i]
        ]
    )


def create_base_optimization_model(game: PricingGame, obs, forecast_grid, include_future_articles=True):
    num_products = game.num_products
    black_prices = obs["black_prices"]
    cw = obs["cw"]
    cogs = obs["cogs"]
    residual_value = obs["residual_value"]
    article_season_start = obs["article_season_start"]
    article_season_end = obs["article_season_end"]
    shipment_costs = obs["shipment_costs"]
    stocks = obs["stocks"]
    num_discounts = forecast_grid.shape[1]  # number of discounts
    discount_range = np.linspace(0, 0.7, num_discounts)

    model = pulp.LpProblem("Revenue_Optimization", pulp.LpMaximize)
    num_weeks = game.num_weeks
    # add variables
    discounts = pulp.LpVariable.dicts(
        "discount",
        [
            (i, w, j)
            for i in range(num_products)
            for w in range(cw, num_weeks)
            for j in range(num_discounts)
            if is_in_optimization(cw, w, article_season_start, article_season_end, include_future_articles)[i]
        ],
        lowBound=0,
        cat="Binary",
    )
    sales_quantity = pulp.LpVariable.dicts(
        "sales_quantity",
        [
            (i, w, j)
            for i in range(num_products)
            for w in range(cw, num_weeks)
            for j in range(num_discounts)
            if is_in_optimization(cw, w, article_season_start, article_season_end, include_future_articles)[i]
        ],
        lowBound=0,
        cat="Continous",
    )
    stock_quantity = pulp.LpVariable.dicts(
        "stock_quantity",
        [(i, w) for i in range(num_products) for w in ([cw - 1] + list(range(cw, num_weeks)))],
        lowBound=0,
        cat="Continous",
    )

    weekly_profit = pulp.LpVariable.dicts(
        "weekly_profit",
        [
            (i, w)
            for i in range(num_products)
            for w in range(cw, num_weeks)
            if is_in_optimization(cw, w, article_season_start, article_season_end, include_future_articles)[i]
        ],
        lowBound=0,
        cat="Continous",
    )
    weekly_revenue = pulp.LpVariable.dicts(
        "weekly_revenue",
        [
            (i, w)
            for i in range(num_products)
            for w in range(cw, num_weeks)
            if is_in_optimization(cw, w, article_season_start, article_season_end, include_future_articles)[i]
        ],
        lowBound=0,
        cat="Continous",
    )

    total_revenue = pulp.LpVariable("revenue", cat="Continuous")
    residual_revenue = pulp.LpVariable("residual_revenue", cat="Continuous")
    future_profit = pulp.LpVariable("profit", cat="Continuous")
    residual_profit = pulp.LpVariable("residual_profit", cat="Continuous")

    # add constraints
    model += total_revenue == pulp.lpSum(
        [
            weekly_revenue[i, w]
            for i in range(num_products)
            for w in range(cw, 52)
            if is_in_optimization(cw, w, article_season_start, article_season_end, include_future_articles)[i]
        ]
    )

    # add residual value
    model += residual_revenue == pulp.lpSum(
        [stock_quantity[(i, 51)] * residual_value[i] for i in range(num_products) if (cw >= article_season_start)[i]]
    )

    model += future_profit == pulp.lpSum(
        [
            weekly_profit[i, w]
            for i in range(num_products)
            for w in range(cw, 52)
            if is_in_optimization(cw, w, article_season_start, article_season_end, include_future_articles)[i]
        ]
    )
    model += residual_profit == pulp.lpSum(
        [
            stock_quantity[(i, 51)] * (residual_value[i] - cogs[i])
            for i in range(num_products)
            if (cw >= article_season_start)[i]
        ]
    )

    for i in range(num_products):
        model += stock_quantity[i, cw - 1] == stocks[i]
        for w in range(cw, 52):
            if is_in_optimization(cw, w, article_season_start, article_season_end, include_future_articles)[i]:
                model += pulp.lpSum([discounts[(i, w, j)] for j in range(num_discounts)]) == 1
                model += weekly_revenue[i, w] == pulp.lpSum(
                    [
                        sales_quantity[(i, w, j)] * (black_prices[i] * (1 - discount_range[j]))
                        for j in range(num_discounts)
                        if is_in_optimization(cw, w, article_season_start, article_season_end, include_future_articles)[
                            i
                        ]
                    ]
                )

                model += weekly_profit[i, w] == pulp.lpSum(
                    [
                        sales_quantity[(i, w, j)]
                        * (black_prices[i] * (1 - discount_range[j]) - cogs[i] - shipment_costs)
                        for j in range(num_discounts)
                        if is_in_optimization(cw, w, article_season_start, article_season_end, include_future_articles)[
                            i
                        ]
                    ]
                )

                for j in range(num_discounts):
                    model += forecast_grid[i][j] * discounts[(i, w, j)] >= sales_quantity[i, w, j]
                model += stock_quantity[i, w - 1] >= pulp.lpSum(sales_quantity[i, w, j] for j in range(num_discounts))
                model += stock_quantity[i, w] == stock_quantity[i, w - 1] - pulp.lpSum(
                    sales_quantity[i, w, j] for j in range(num_discounts)
                )
            else:
                model += stock_quantity[i, w] == stock_quantity[i, w - 1]

    model_dict = {
        "model": model,
        "discounts": discounts,
        "sales": sales_quantity,
        "stocks": stock_quantity,
        "inseason_revenue": total_revenue,
        "residual_revenue": residual_revenue,
        "inseason_profit": future_profit,
        "residual_profit": residual_profit,
        "weekly_profit": weekly_profit,
        "weekly_revenue": weekly_revenue,
    }
    # add_profit_penalty_objective(model, total_revenue, residual_revenue, game, future_profit, residual_profit)

    return model_dict
