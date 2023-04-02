import plotly.graph_objects as go


class GameState:
    def __init__(self, profits, revenues, sales, sdrs, discounts):
        self.profits = profits
        self.revenues = revenues
        self.sales = sales
        self.sdrs = sdrs
        self.discounts = discounts

    def visualize(self):
        summed_revenue = [r.sum() for r in self.revenues]
        summed_profit = [p.sum() for p in self.profits]
        x = list(range(len(self.profits)))
        trace1 = go.Scatter(x=x, y=summed_revenue, mode="lines", name="Revenue")
        trace2 = go.Scatter(x=x, y=summed_profit, mode="lines", name="Profits")

        fig = go.Figure(data=[trace1, trace2])
        fig.update_layout(title="Revenue and Profit", xaxis_title="Calendar Week", yaxis_title="Euros")
        fig.update_traces(line=dict(width=2))
        fig.show()

        fig = go.Figure(data=go.Scatter(x=x, y=self.sdrs, mode="lines"))
        fig.update_layout(title="SDR", xaxis_title="Calendar Week", yaxis_title="SDR")
        fig.update_traces(line=dict(width=2))
        fig.show()
