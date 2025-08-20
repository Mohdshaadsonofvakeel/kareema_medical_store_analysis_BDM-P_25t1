# 🏥 Optimizing Pharmacy Inventory: A Four-Month Analysis of Seasonal Trends and Demand Patterns

[![Project Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)](https://github.com/yourusername/pharmacy-inventory-optimization)
[![Score](https://img.shields.io/badge/Score-97%2F100-brightgreen?style=flat-square)](https://github.com/yourusername/pharmacy-inventory-optimization)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0.3-orange?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15.0-purple?style=flat-square&logo=plotly)](https://plotly.com/)

> **BDM Capstone Project Final Submission**  
> **Student**: Mohd Shad (23f3004148@ds.study.iitm.ac.in)  
> **Institution**: IIT Madras Online BS Degree Program  
> **Duration**: September 2024 – February 2025  
> **Project Score**: 97/100 (Approved)

---

## 📋 Table of Contents

- [Executive Summary](#-executive-summary)
- [Business Problem](#-business-problem)
- [Technical Stack](#-technical-stack)
- [Data Analysis](#-data-analysis)
- [Key Visualizations](#-key-visualizations)
- [Results & Findings](#-results--findings)
- [Recommendations](#-recommendations)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Research Applications](#-research-applications)
- [Contact](#-contact)

---

## 🎯 Executive Summary

This comprehensive analysis investigates critical inventory management challenges faced by **Kareema Medical Store** during September 2024 to February 2025. The study addresses two fundamental problems:

1. **Unpredictable demand patterns** leading to frequent stockouts
2. **Complex pricing variations** affecting profitability margins

### Key Achievements
- **311 SKUs** analyzed across **15 therapeutic categories**
- **ARIMA(6,1,6)** forecasting model with **21.67% MAPE**
- **₹93,485** total revenue analyzed
- **62.34%** revenue concentration in top 5 categories
- **132%** December demand spike successfully predicted

---

## 🏢 Business Problem

### Primary Challenges

#### Problem 1: Demand Volatility & Inventory Inefficiency
- Seasonal health trends causing unpredictable demand spikes
- Frequent stockouts of critical medicines (Antibiotics, NSAIDs)
- Working capital locked in slow-moving inventory
- **₹3,400** in lost sales due to stockouts

#### Problem 2: Pricing Variations & Profitability
- Fluctuating distributor pricing complicating purchase decisions
- Similar medicines with varying costs (e.g., "Zerodol SP" vs "Zerodol")
- Premium variants showing 28-35% higher purchase values but only 20-25% higher revenue
- **₹12,500** potential working capital optimization

---

## 🛠 Technical Stack

```python
# Core Dependencies
pandas==2.0.3          # Data manipulation and analysis
plotly==5.15.0         # Interactive visualizations
openpyxl==3.1.2        # Excel file handling
statsmodels==0.13.1    # Time series forecasting
numpy>=1.21.0          # Numerical computations
matplotlib>=3.5.0      # Static plotting
seaborn>=0.11.0        # Statistical visualizations
```

### Development Environment
- **Python**: 3.8+
- **Platform**: Google Colab
- **Data Source**: Marg ERP System
- **Analysis Period**: September 2024 - February 2025

---

## 📊 Data Analysis

### Dataset Overview
- **Rows**: 311 SKUs
- **Columns**: 15 attributes
- **Time Period**: 6 months (Sep 2024 - Feb 2025)
- **Categories**: 15 therapeutic classifications

### Key Metrics
| Metric | Value |
|--------|-------|
| Total Revenue | ₹93,485.37 |
| Total Purchase Value | ₹280,456.12 |
| Average Revenue/SKU | ₹301.57 |
| Total Issue Quantity | 33,883 units |
| Total Dump Quantity | 87,578 units |

### Data Preprocessing Steps
1. **Missing Value Treatment**: Replaced missing values with appropriate defaults
2. **Numerical Standardization**: Converted to consistent formats
3. **Category Mapping**: Organized into 15 therapeutic classifications
4. **Validation**: Cross-verified against physical inventory

---

## 📈 Key Visualizations
### 📊 Key Visualizations

#### 1. **Pareto Analysis: Revenue Distribution**
![Pareto Analysis](https://github.com/Mohdshaadsonofvakeel/kareema_medical_store_analysis_BDM-P_25t1/raw/main/charts/pareto_analysis.png)

#### 2. **Stockout Risk Heatmap**
![Stockout Heatmap](https://github.com/Mohdshaadsonofvakeel/kareema_medical_store_analysis_BDM-P_25t1/raw/main/charts/stockout_heatmap.png)

#### 3. **ARIMA Forecast**
![ARIMA Forecast](https://github.com/Mohdshaadsonofvakeel/kareema_medical_store_analysis_BDM-P_25t1/raw/main/charts/arima_forecast.png)

#### 4. **ABC Analysis Detailed**
![ABC Analysis](https://github.com/Mohdshaadsonofvakeel/kareema_medical_store_analysis_BDM-P_25t1/raw/main/charts/abc_analysis_detailed.png)

#### 5. **Class Metrics Comparison**
![Class Metrics](https://github.com/Mohdshaadsonofvakeel/kareema_medical_store_analysis_BDM-P_25t1/raw/main/charts/class_metrics_comparison.png)

#### 6. **Medicine Price Analysis**
![Medicine Price Analysis](https://github.com/Mohdshaadsonofvakeel/kareema_medical_store_analysis_BDM-P_25t1/raw/main/charts/medicine_price_analysys.png)

### 1. Pareto Analysis: Revenue Distribution

```python
import pandas as pd
import plotly.graph_objects as go

# Calculate revenue by category
cat_revenue = df.groupby('Therapeutic Tag')['Revenue'].sum().sort_values(ascending=False)
total_revenue = cat_revenue.sum()
cat_revenue_pct = cat_revenue / total_revenue * 100
cum_pct = cat_revenue_pct.cumsum()

# Create Pareto Chart
fig = go.Figure()
fig.add_trace(go.Bar(x=cat_revenue.index, y=cat_revenue.values, name='Revenue'))
fig.add_trace(go.Scatter(x=cat_revenue.index, y=cum_pct, mode='lines+markers', 
                        name='Cumulative %', yaxis='y2'))
```

**Key Insights:**
- **5 categories** generate **62.34%** of total revenue
- **Antibiotics** lead with **19.02%** contribution (₹6,419.47)
- Classic Pareto distribution validates focused management approach

### 2. Stockout Risk Heat Map

```python
# Create stockout risk measure
risk_df['Stockout Risk'] = np.where(
    risk_df['CLOSING_QTY'] < 0, 3,  # High risk
    np.where(risk_df['CLOSING_QTY'] < risk_df['ISSUE_QTY'] * 0.1, 2,  # Medium-high
    np.where(risk_df['CLOSING_QTY'] < risk_df['ISSUE_QTY'] * 0.3, 1, 0)))  # Low
```

**Risk Classification:**
- **High Risk (3)**: Negative closing quantities
- **Medium-High Risk (2)**: < 10% of issue quantity
- **Medium Risk (1)**: < 30% of issue quantity  
- **Low Risk (0)**: > 30% of issue quantity

### 3. ARIMA Time Series Forecasting

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(ts_data['Sales'], order=(6, 1, 6))
model_fit = model.fit()

# 8-week forecast
forecast = model_fit.forecast(steps=8)
```

**Model Performance:**
- **MAPE**: 21.67% for high-volatility categories
- **ADF Test**: p-value < 0.05 (stationary after differencing)
- **Forecast Horizon**: 8 weeks with 95% confidence intervals

### 4. ABC Classification Analysis

**Class Distribution:**
| Class | SKUs | SKU % | Revenue % | Waste % | Stockouts |
|-------|------|-------|-----------|---------|-----------|
| A | 83 | 26.8% | 64.6% | 1.5% | 16 |
| B | 76 | 24.5% | 25.3% | 5.9% | 7 |
| C | 151 | 48.7% | 10.2% | 5.4% | 2 |

---

## 🔍 Results & Findings

### Problem 1: Demand Patterns & Stockouts

#### Revenue Concentration Analysis
- **Top 5 therapeutic categories** drive majority of business
- **Antibiotics** show highest volatility (SD: 96 units)
- **NSAIDs** exhibit seasonal correlation with flu outbreaks
- **December spike**: 132% week-over-week increase (₹45,000 → ₹104,705)

#### Stockout Risk Assessment
- **Antibiotics**: 9.0% high-risk scenarios despite being top revenue generator
- **NSAIDs**: 9.8% high-risk situations with higher volatility
- **Inverse relationship**: High-revenue categories face highest stockout risks

#### Temporal Patterns
- **Winter surge**: Dramatic December peak followed by January decline
- **Seasonal transitions**: Rapid demand changes challenge inventory planning
- **Predictable volatility**: ARIMA successfully captures seasonal patterns

### Problem 2: Pricing & Profitability

#### Price Structure Analysis
- **Three-tier pricing**: Premium (₹800-1200), Mid-tier (₹400-700), Standard (<₹300)
- **Median purchase value**: ₹141.31
- **Median issue value**: ₹188.41
- **Base margin structure**: Varies by category and price point

#### Working Capital Impact
- **Premium variants**: 40% longer inventory holding periods
- **Highest price quartile**: 2.5x more working capital per revenue unit
- **Optimization potential**: ₹12,500 in freed capital for Antibiotics alone

---

## 💡 Recommendations

### 1. Demand Forecasting Enhancement
- **Implement ARIMA modeling** for high-revenue categories
- **Seasonal preparation**: 3x normal stock for December Antibiotic demand
- **Weekly pattern recognition**: 12-15% higher demand in weeks 1 & 3 of each month

### 2. Inventory Priority Realignment
- **Focus on Class A items**: 64.6% revenue with minimal waste
- **Risk-based stocking**: Align inventory protection with revenue contribution
- **Safety stock optimization**: 17% reduction possible while maintaining service levels

### 3. Pricing Strategy Refinement
- **Therapeutic alternatives analysis**: Balance premium pricing with turnover rates
- **Cost-effectiveness ratios**: (Revenue - Purchase_Cost) / Issue_Quantity
- **Dynamic pricing**: Adjust based on demand predictability coefficients

### 4. Operational Improvements
- **Prevent 73% of stockouts** using forecasting model
- **Reduce working capital** by ₹12,500 in high-volume categories
- **Minimize waste**: Target Class B & C optimization (5.9% and 5.4% waste respectively)

---

## 🚀 Installation & Usage

### Prerequisites
```bash
python>=3.8
jupyter notebook or Google Colab
```

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/pharmacy-inventory-optimization.git
cd pharmacy-inventory-optimization

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Quick Start
```python
# Load and analyze data
import pandas as pd
import plotly.graph_objects as go

# Load dataset
df = pd.read_excel('KAREEMA_MEDICAL_STORE.xlsx')

# Run Pareto analysis
from analysis.pareto_analysis import generate_pareto_chart
pareto_fig = generate_pareto_chart(df)
pareto_fig.show()

# Time series forecasting
from analysis.forecasting import arima_forecast
forecast_results = arima_forecast(df, category='Antibiotic', periods=8)
```

---

## 📁 Project Structure

```
pharmacy-inventory-optimization/
├── data/
│   ├── KAREEMA_MEDICAL_STORE.xlsx
│   └── time_series_data.xlsx
├── analysis/
│   ├── pareto_analysis.py
│   ├── stockout_risk.py
│   ├── abc_classification.py
│   └── forecasting.py
├── visualizations/
│   ├── pareto_chart.html
│   ├── heatmap_stockout.html
│   ├── weekly_trends.html
│   └── arima_forecast.html
├── reports/
│   ├── final_submission.pdf
│   ├── mid_term_report.pdf
│   └── proposal.pdf
├── notebooks/
│   └── Karema_medical_store_charts.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🎓 Research Applications

### Alignment with Research Internship Goals

This project provides substantial foundation for research in:

#### 1. **Business Analytics & Inventory Optimization**
- Data-driven inventory management methodologies
- Demand forecasting in healthcare supply chains
- ABC classification optimization techniques

#### 2. **Time Series Analysis & Forecasting**
- ARIMA modeling for seasonal demand patterns
- Statistical validation techniques (ADF testing)
- Confidence interval analysis for business decisions

#### 3. **Healthcare Operations Research**
- Pharmaceutical inventory challenges
- Seasonal demand modeling in healthcare
- Risk assessment frameworks for medical supplies

### Future Research Opportunities

1. **Machine Learning Enhancement**
   - LSTM networks for sequential demand prediction
   - Ensemble methods combining ARIMA with ML models
   - Real-time adaptive forecasting systems

2. **Advanced Analytics Applications**
   - Dynamic pricing optimization models
   - Supply chain network analysis
   - Patient health trend correlation studies

3. **Decision Support Systems**
   - Automated inventory management platforms
   - Integrated ERP optimization solutions
   - Predictive analytics dashboards

---

## 📚 Publications & Documentation

### Academic Contributions
- **Comprehensive methodology** for small-scale pharmacy inventory optimization
- **Validated forecasting approach** achieving 21.67% MAPE
- **Practical implementation framework** for resource-constrained environments

### Conference Presentations
*Ready for submission to:*
- Operations Research Society conferences
- Healthcare analytics symposiums
- Business intelligence workshops

---

## 🏆 Project Achievements

### Academic Recognition
- **Final Score**: 97/100
- **Viva Voce**: Passed
- **Rubrics Excellence**: Business Brief, Analysis, Interpretation, Recommendations

### Technical Milestones
- **Real-world data collection** from Marg ERP system
- **Multi-dimensional analysis** combining statistical and visual methods
- **Actionable insights** with quantified business impact
- **Reproducible methodology** for similar business contexts

---

## 🤝 Connect & Collaborate

### Research Collaboration Interests
- **Inventory Optimization**: Supply chain analytics and forecasting
- **Healthcare Analytics**: Medical inventory management systems  
- **Business Intelligence**: Data-driven decision support systems
- **Time Series Analysis**: Demand forecasting methodologies

### Contact Information
- **Email**: 23f3004148@ds.study.iitm.ac.in
- **LinkedIn**: [Connect for research opportunities]
- **Institution**: IIT Madras Online BS Degree Program
- **Research Focus**: Data Science & Business Analytics

---

## 📄 License & Citation

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation
```bibtex
@misc{shad2025pharmacy,
  title={Optimizing Pharmacy Inventory: A Four-Month Analysis of Seasonal Trends and Demand Patterns},
  author={Shad, Mohd},
  year={2025},
  institution={Indian Institute of Technology Madras},
  type={Capstone Project},
  note={BDM Final Submission - Score: 97/100}
}
```

---

## 🙏 Acknowledgments

- **Kareema Medical Store** & **Mohd Umar** for data access and business insights
- **Abrar Memorial Hospital** for facilitating research collaboration
- **IIT Madras Faculty** for academic guidance and methodology support
- **BDM Capstone Program** for providing real-world analysis opportunities

---

*This project demonstrates the practical application of data science techniques to solve real-world business challenges in healthcare inventory management. The methodologies and insights presented here contribute to the growing field of healthcare operations research and provide a foundation for future academic and industry collaborations.*

**Project Status**: ✅ **Completed Successfully** | **Ready for Research Applications**
