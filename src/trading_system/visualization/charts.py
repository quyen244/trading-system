"""
Beautiful Visualization Dashboard for Trading System

Creates stunning, professional charts for backtest analysis:
1. Equity Curve Chart
2. Drawdown Analysis Chart
3. Win/Loss Distribution Chart
4. Monthly Returns Heatmap
5. Strategy Comparison Chart

Author: Trading System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from typing import Dict, List, Optional
from datetime import datetime
from trading_system.backtesting.trade import Trade
import warnings
from trading_system.utils.logger import setup_logger

logger = setup_logger('charts')
warnings.filterwarnings('ignore')

# Set beautiful style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TradingVisualizer:
    """
    Create beautiful, professional visualizations for trading analysis.
    
    Example usage:
        viz = TradingVisualizer()
        viz.plot_equity_curve(equity_series, title="My Strategy")
        viz.plot_drawdown(equity_series)
        viz.plot_monthly_returns(equity_series)
        viz.show()
    """
    
    def __init__(self, style: str = 'dark'):
        """
        Initialize visualizer.
        
        Args:
            style: 'dark' or 'light' theme
        """
        self.style = style
        self.figures = []
        
        # Set theme
        if style == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#1a1a1a'
            self.grid_color = '#333333'
            self.text_color = '#ffffff'
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.bg_color = '#ffffff'
            self.grid_color = '#e0e0e0'
            self.text_color = '#000000'
    
    def plot_equity_curve(
        self,
        equity_series: pd.Series,
        title: str = "Equity Curve",
        benchmark: Optional[pd.Series] = None,
        figsize: tuple = (14, 7)
    ):
        """
        Plot beautiful equity curve with optional benchmark.
        
        Args:
            equity_series: Equity curve data
            title: Chart title
            benchmark: Optional benchmark series (e.g., buy & hold)
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
        ax.set_facecolor(self.bg_color)
        
        # Plot equity curve
        ax.plot(equity_series.index, equity_series.values, 
                linewidth=2.5, label='Strategy', color='#00ff88', alpha=0.9)
        
        # Plot benchmark if provided
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark.values,
                   linewidth=2, label='Benchmark', color='#ff6b6b', 
                   alpha=0.7, linestyle='--')
        
        # Fill area under curve
        ax.fill_between(equity_series.index, equity_series.values,
                        alpha=0.2, color='#00ff88')
        
        # Styling
        ax.set_title(title, fontsize=18, fontweight='bold', 
                    color=self.text_color, pad=20)
        ax.set_xlabel('Date', fontsize=12, color=self.text_color)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12, color=self.text_color)
        ax.grid(True, alpha=0.3, color=self.grid_color)
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add statistics box
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
        max_value = equity_series.max()
        min_value = equity_series.min()
        
        stats_text = f'Total Return: {total_return:.2f}%\n'
        stats_text += f'Max Value: ${max_value:,.0f}\n'
        stats_text += f'Min Value: ${min_value:,.0f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=self.bg_color, 
                        alpha=0.8, edgecolor='#00ff88'),
               color=self.text_color)
        
        plt.tight_layout()
        self.figures.append(fig)
        
        return fig
    
    def plot_drawdown(
        self,
        equity_series: pd.Series,
        title: str = "Drawdown Analysis",
        figsize: tuple = (14, 6)
    ):
        """
        Plot drawdown chart showing underwater equity.
        
        Args:
            equity_series: Equity curve data
            title: Chart title
            figsize: Figure size
        """
        # Calculate drawdown
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                       facecolor=self.bg_color,
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Equity with running max
        ax1.set_facecolor(self.bg_color)
        ax1.plot(equity_series.index, equity_series.values,
                linewidth=2, label='Equity', color='#00ff88')
        ax1.plot(running_max.index, running_max.values,
                linewidth=1.5, label='Running Max', color='#ffd700',
                linestyle='--', alpha=0.7)
        ax1.fill_between(equity_series.index, equity_series.values, running_max.values,
                        alpha=0.3, color='#ff6b6b')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=11, color=self.text_color)
        ax1.grid(True, alpha=0.3, color=self.grid_color)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot 2: Drawdown percentage
        ax2.set_facecolor(self.bg_color)
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                        alpha=0.6, color='#ff6b6b')
        ax2.plot(drawdown.index, drawdown.values,
                linewidth=2, color='#ff4444')
        ax2.set_ylabel('Drawdown (%)', fontsize=11, color=self.text_color)
        ax2.set_xlabel('Date', fontsize=11, color=self.text_color)
        ax2.grid(True, alpha=0.3, color=self.grid_color)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        # Add max drawdown annotation
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax2.annotate(f'Max DD: {max_dd:.2f}%',
                    xy=(max_dd_date, max_dd),
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor=self.bg_color, alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='#ff4444'),
                    fontsize=10, color=self.text_color)
        
        fig.suptitle(title, fontsize=18, fontweight='bold',
                    color=self.text_color, y=0.98)
        
        plt.tight_layout()
        self.figures.append(fig)
        
        return fig
    
    def plot_trade_distribution(
        self,
        trades: List[Dict],
        title: str = "Trade Distribution",
        figsize: tuple = (14, 6)
    ):
        """
        Plot win/loss distribution and statistics.
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            title: Chart title
            figsize: Figure size
        """
        if not trades:
            print("No trades to plot")
            return None

        pnls = [trade['net_pnl'] for trade in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor=self.bg_color)
        
        # Plot 1: Histogram
        ax1.set_facecolor(self.bg_color)
        ax1.hist(wins, bins=20, alpha=0.7, color='#00ff88', label='Wins', edgecolor='black')
        ax1.hist(losses, bins=20, alpha=0.7, color='#ff6b6b', label='Losses', edgecolor='black')
        ax1.set_xlabel('P&L ($)', fontsize=11, color=self.text_color)
        ax1.set_ylabel('Frequency', fontsize=11, color=self.text_color)
        ax1.set_title('P&L Distribution', fontsize=14, fontweight='bold', color=self.text_color)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, color=self.grid_color)
        ax1.axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.5)
        
        # Plot 2: Statistics
        ax2.set_facecolor(self.bg_color)
        ax2.axis('off')
        
        # Calculate statistics
        total_trades = len(pnls)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        total_pnl = sum(pnls)
        
        stats_text = f"""
        📊 TRADE STATISTICS
        {'='*40}
        
        Total Trades:        {total_trades}
        Winning Trades:      {len(wins)} ({win_rate:.1f}%)
        Losing Trades:       {len(losses)} ({100-win_rate:.1f}%)
        
        Average Win:         ${avg_win:,.2f}
        Average Loss:        ${avg_loss:,.2f}
        Profit Factor:       {profit_factor:.2f}
        
        Total P&L:           ${total_pnl:,.2f}
        Best Trade:          ${max(pnls):,.2f}
        Worst Trade:         ${min(pnls):,.2f}
        """
        
        ax2.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', color=self.text_color,
                bbox=dict(boxstyle='round', facecolor=self.bg_color,
                         alpha=0.8, edgecolor='#00ff88', linewidth=2))
        
        fig.suptitle(title, fontsize=18, fontweight='bold',
                    color=self.text_color, y=0.98)
        
        plt.tight_layout()
        self.figures.append(fig)
        
        return fig
    
    def plot_monthly_returns(
        self,
        equity_series: pd.Series,
        title: str = "Monthly Returns Heatmap",
        figsize: tuple = (14, 8)
    ):
        """
        Plot monthly returns heatmap.
        
        Args:
            equity_series: Equity curve data
            title: Chart title
            figsize: Figure size
        """
        # Calculate monthly returns
        monthly_equity = equity_series.resample('M').last()
        monthly_returns = monthly_equity.pct_change() * 100
        
        # Create pivot table
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot = monthly_returns_df.pivot(index='Month', columns='Year', values='Return')
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.index = [month_names[i-1] for i in pivot.index]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
        ax.set_facecolor(self.bg_color)
        
        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Return (%)'},
                   linewidths=1, linecolor=self.grid_color,
                   ax=ax, vmin=-10, vmax=10)
        
        ax.set_title(title, fontsize=18, fontweight='bold',
                    color=self.text_color, pad=20)
        ax.set_xlabel('Year', fontsize=12, color=self.text_color)
        ax.set_ylabel('Month', fontsize=12, color=self.text_color)
        
        plt.tight_layout()
        self.figures.append(fig)
        
        return fig
    
    def plot_strategy_comparison(
        self,
        equity_curves: Dict[str, pd.Series],
        title: str = "Strategy Comparison",
        figsize: tuple = (14, 8)
    ):
        """
        Plot multiple strategies for comparison.
        
        Args:
            equity_curves: Dictionary of {strategy_name: equity_series}
            title: Chart title
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                       facecolor=self.bg_color,
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot 1: Equity curves
        ax1.set_facecolor(self.bg_color)
        
        colors = ['#00ff88', '#ff6b6b', '#4ecdc4', '#ffd700', '#ff9ff3', '#54a0ff']
        
        for i, (name, equity) in enumerate(equity_curves.items()):
            # Normalize to start at 100
            normalized = (equity / equity.iloc[0]) * 100
            color = colors[i % len(colors)]
            ax1.plot(normalized.index, normalized.values,
                    linewidth=2.5, label=name, color=color, alpha=0.8)
        
        ax1.set_ylabel('Normalized Value', fontsize=11, color=self.text_color)
        ax1.set_title(title, fontsize=18, fontweight='bold',
                     color=self.text_color, pad=15)
        ax1.grid(True, alpha=0.3, color=self.grid_color)
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax1.axhline(y=100, color='white', linestyle='--', linewidth=1, alpha=0.5)
        
        # Plot 2: Returns comparison bar chart
        ax2.set_facecolor(self.bg_color)
        
        returns = {}
        for name, equity in equity_curves.items():
            total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
            returns[name] = total_return
        
        names = list(returns.keys())
        values = list(returns.values())
        bar_colors = [colors[i % len(colors)] for i in range(len(names))]
        
        bars = ax2.bar(names, values, color=bar_colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Total Return (%)', fontsize=11, color=self.text_color)
        ax2.set_xlabel('Strategy', fontsize=11, color=self.text_color)
        ax2.grid(True, alpha=0.3, axis='y', color=self.grid_color)
        ax2.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9, color=self.text_color)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        self.figures.append(fig)
        
        return fig

    def plot_candlestick_with_trades(
        self,
        df: pd.DataFrame,
        trades: List[Trade],
        symbol: str = "Symbol",
        title: str = "Price Chart with Trades"
    ):
        """
        Create a professional Plotly candlestick chart with trade annotations.
        """
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))

        # Add Trades
        for trade in trades:
            # Entry Marker
            fig.add_trace(go.Scatter(
                x=[trade.entry_time],
                y=[trade.entry_price],
                mode='markers',
                marker=dict(symbol='circle', size=10, color='blue' if trade.side == 'long' else 'orange'),
                name=f"Entry ({trade.side})",
                showlegend=False
            ))
            
            # Exit Marker
            if trade.exit_time:
                fig.add_trace(go.Scatter(
                    x=[trade.exit_time],
                    y=[trade.exit_price],
                    mode='markers',
                    marker=dict(symbol='x', size=10, color='white'),
                    name="Exit",
                    showlegend=False
                ))
            
            # SL/TP Lines (blurred/transparent)
            if hasattr(trade, 'stop_loss') and trade.stop_loss:
                # Calculate a duration to show the markers/lines
                # For simplicity, we can just show them at entry
                fig.add_trace(go.Scatter(
                    x=[trade.entry_time, trade.exit_time or df.index[-1]],
                    y=[trade.stop_loss, trade.stop_loss],
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0.3)', width=2, dash='dash'),
                    name="Stop Loss",
                    showlegend=False
                ))
            
            if hasattr(trade, 'take_profit') and trade.take_profit:
                fig.add_trace(go.Scatter(
                    x=[trade.entry_time, trade.exit_time or df.index[-1]],
                    y=[trade.take_profit, trade.take_profit],
                    mode='lines',
                    line=dict(color='rgba(0, 255, 0, 0.3)', width=2, dash='dash'),
                    name="Take Profit",
                    showlegend=False
                ))

        fig.update_layout(
            title=title,
            yaxis_title=f"{symbol} Price",
            xaxis_title="Date",
            template="plotly_dark",
            height=700,
            xaxis_rangeslider_visible=False
        )

        return fig

    def show(self):
        """Display all created figures."""
        plt.show()
    
    def save_all(self, prefix: str = "chart", dpi: int = 300):
        """
        Save all figures to files.
        
        Args:
            prefix: Filename prefix
            dpi: Image resolution
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, fig in enumerate(self.figures):
            filename = f"{prefix}_{i+1}_{timestamp}.png"
            fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                       facecolor=self.bg_color)
            print(f"Saved: {filename}")


if __name__ == "__main__":
    print("Trading Visualization Dashboard")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    from trading_system.visualization.charts import TradingVisualizer
    
    # Create visualizer
    viz = TradingVisualizer(style='dark')
    
    # Plot equity curve
    viz.plot_equity_curve(equity_series, title="My Strategy")
    
    # Plot drawdown
    viz.plot_drawdown(equity_series)
    
    # Plot trade distribution
    viz.plot_trade_distribution(trades)
    
    # Plot monthly returns
    viz.plot_monthly_returns(equity_series)
    
    # Plot strategy comparison
    viz.plot_strategy_comparison(equity_curves_dict)
    
    # Show all charts
    viz.show()
    
    # Or save them
    viz.save_all(prefix="backtest_results")
    """)
