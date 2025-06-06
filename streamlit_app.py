import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import yfinance as yf  # For real economic data
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="EconoPredict - لوحة تحليل الاقتصاد التركي",
    page_icon="📈",
    layout="wide"
)

# نظام متعدد اللغات
def load_translations():
    return {
        'ar': {
            'title': 'EconoPredict - توقع الناتج المحلي والتضخم في تركيا',
            'gdp_growth': 'نمو الناتج المحلي الإجمالي (%)',
            'inflation_rate': 'معدل التضخم (%)',
            'dashboard': 'لوحة التحكم',
            'historical_data': 'البيانات التاريخية',
            'forecasts': 'التنبؤات',
            'gdp_forecast': 'تنبؤ نمو الناتج المحلي',
            'inflation_forecast': 'تنبؤ معدل التضخم',
            'model_comparison': 'مقارنة النماذج',
            'latest_data': 'أحدث البيانات',
            'select_model': 'اختر نموذج التنبؤ',
            'select_indicator': 'اختر المؤشر',
            'download_report': 'تحميل التقرير',
            'run_analysis': 'تشغيل التحليل',
            'analysis_in_progress': 'جاري التحليل...',
            'completed_successfully': 'تم بنجاح',
            'report_generated': 'تم إنشاء التقرير',
            'model_performance': 'أداء النماذج',
            'confidence_interval': 'فترة الثقة',
            'historical_forecast': 'المقارنة التاريخية',
            'feature_importance': 'أهمية العوامل'
        },
        'en': {
            'title': 'EconoPredict - Turkish Economic Dashboard',
            'gdp_growth': 'GDP Growth (%)',
            'inflation_rate': 'Inflation Rate (%)',
            'dashboard': 'Dashboard',
            'historical_data': 'Historical Data',
            'forecasts': 'Forecasts',
            'gdp_forecast': 'GDP Growth Forecast',
            'inflation_forecast': 'Inflation Rate Forecast',
            'model_comparison': 'Model Comparison',
            'latest_data': 'Latest Data',
            'select_model': 'Select Forecast Model',
            'select_indicator': 'Select Indicator',
            'download_report': 'Download Report',
            'run_analysis': 'Run Analysis',
            'analysis_in_progress': 'Analysis in progress...',
            'completed_successfully': 'Completed successfully',
            'report_generated': 'Report generated',
            'model_performance': 'Model Performance',
            'confidence_interval': 'Confidence Interval',
            'historical_forecast': 'Historical Forecast',
            'feature_importance': 'Feature Importance'
        },
        'tr': {
            'title': 'EconoPredict - Türkiye Ekonomi Panosu',
            'gdp_growth': 'GSYİH Büyümesi (%)',
            'inflation_rate': 'Enflasyon Oranı (%)',
            'dashboard': 'Kontrol Paneli',
            'historical_data': 'Tarihsel Veri',
            'forecasts': 'Tahminler',
            'gdp_forecast': 'GSYİH Büyüme Tahmini',
            'inflation_forecast': 'Enflasyon Oranı Tahmini',
            'model_comparison': 'Model Karşılaştırması',
            'latest_data': 'Son Veriler',
            'select_model': 'Tahmin Modeli Seçin',
            'select_indicator': 'Gösterge Seçin',
            'download_report': 'Raporu İndir',
            'run_analysis': 'Analizi Çalıştır',
            'analysis_in_progress': 'Analiz devam ediyor...',
            'completed_successfully': 'Başarıyla tamamlandı',
            'report_generated': 'Rapor oluşturuldu',
            'model_performance': 'Model Performansı',
            'confidence_interval': 'Güven Aralığı',
            'historical_forecast': 'Tarihsel Tahmin',
            'feature_importance': 'Özellik Önemi'
        }
    }

translations = load_translations()

# اختيار اللغة
lang = st.sidebar.selectbox("🌍 اللغة / Language", 
                           list(translations.keys()),
                           format_func=lambda x: {"ar": "العربية", "en": "English", "tr": "Türkçe"}[x])
t = translations[lang]

# عنوان التطبيق
st.title(t['title'])
st.markdown("""
<style>
    .header-style {
        font-size:30px;
        color:#1e3799;
        font-weight:bold;
        padding-bottom:10px;
        border-bottom:2px solid #4a69bd;
        margin-bottom:20px;
    }
    .metric-card {
        background-color:#f8f9fa;
        border-radius:10px;
        padding:15px;
        margin:10px;
        box-shadow:0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow:0 8px 12px rgba(0,0,0,0.15);
    }
    .metric-title {
        font-size:14px;
        color:#6c757d;
        margin-bottom:5px;
    }
    .metric-value {
        font-size:24px;
        font-weight:bold;
        color:#1e3799;
    }
    .stButton>button {
        background-color: #1e3799 !important;
        color: white !important;
        border-radius: 8px;
        padding: 8px 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0c2461 !important;
        transform: scale(1.05);
    }
    .footer {
        text-align: center;
        padding: 15px;
        margin-top: 30px;
        border-top: 1px solid #ddd;
        color: #555;
        font-size: 14px;
    }
    .stProgress > div > div > div {
        background-color: #1e3799;
    }
</style>
""", unsafe_allow_html=True)

# تحميل بيانات حقيقية
@st.cache_data
def load_real_data():
    try:
        # الحصول على بيانات من ياهو فاينانس
        gdp = yf.download('TURGDP=ECI', period="20y")['Close'].dropna()
        inflation = yf.download('TURCPI=ECI', period="20y")['Close'].dropna()
        
        # إنشاء إطار بيانات موحد
        df = pd.DataFrame({
            'Date': gdp.index,
            'GDP Growth': gdp.pct_change().fillna(0) * 100,
            'Inflation': inflation
        })
        df['Year'] = df['Date'].dt.year
        annual_df = df.groupby('Year').mean().reset_index()
        return annual_df.tail(25)  # آخر 25 سنة
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {str(e)}")
        return load_sample_data()

def load_sample_data():
    return pd.DataFrame({
        'Year': list(range(1999, 2024)),
        'GDP Growth': [6.3, 6.1, 5.3, 6.2, 9.4, 8.4, 6.9, 4.7, 0.7,
                       4.7, 8.5, 11.1, 4.8, 8.5, 5.2, 6.1, 3.2, 7.4,
                       0.8, -2.8, 1.8, 11.4, 5.5, 5.6, 4.5],
        'Inflation': [68.9, 54.4, 45.0, 29.7, 18.4, 8.6, 8.2, 9.6, 10.1,
                      8.6, 6.3, 8.6, 6.2, 7.4, 8.2, 7.7, 11.1, 16.3,
                      15.2, 11.8, 14.6, 19.6, 15.2, 64.3, 53.9]
    })

data = load_real_data()

# نماذج التنبؤ
@st.cache_resource
def train_models(data):
    models = {}
    
    # ARIMA لنمو الناتج المحلي
    gdp_data = data['GDP Growth'].values
    arima_gdp = ARIMA(gdp_data, order=(2,1,2)).fit()
    models['ARIMA_GDP'] = arima_gdp
    
    # ARIMA للتضخم
    inflation_data = data['Inflation'].values
    arima_inflation = ARIMA(inflation_data, order=(1,1,1)).fit()
    models['ARIMA_INFLATION'] = arima_inflation
    
    # راندوم فورست
    X = data[['Year']]
    y_gdp = data['GDP Growth']
    y_inflation = data['Inflation']
    
    rf_gdp = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_gdp.fit(X, y_gdp)
    models['RF_GDP'] = rf_gdp
    
    rf_inflation = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_inflation.fit(X, y_inflation)
    models['RF_INFLATION'] = rf_inflation
    
    return models

# تحميل النماذج
models = train_models(data)

# وظائف التنبؤ
def forecast_gdp(model_name, years_ahead=2):
    last_year = data['Year'].max()
    forecast_years = [last_year + i for i in range(1, years_ahead+1)]
    
    if 'ARIMA' in model_name:
        forecast = models['ARIMA_GDP'].forecast(steps=years_ahead)
    else:
        forecast = models['RF_GDP'].predict(np.array(forecast_years).reshape(-1, 1))
    
    return forecast_years, forecast

def forecast_inflation(model_name, years_ahead=2):
    last_year = data['Year'].max()
    forecast_years = [last_year + i for i in range(1, years_ahead+1)]
    
    if 'ARIMA' in model_name:
        forecast = models['ARIMA_INFLATION'].forecast(steps=years_ahead)
    else:
        forecast = models['RF_INFLATION'].predict(np.array(forecast_years).reshape(-1, 1))
    
    return forecast_years, forecast

# قسم البيانات التاريخية
st.subheader(t['historical_data'])
col1, col2 = st.columns(2)

with col1:
    fig1 = px.line(data, x='Year', y='GDP Growth', 
                  title=t['gdp_growth'], markers=True,
                  labels={'GDP Growth': t['gdp_growth'], 'Year': 'السنة'})
    fig1.update_layout(template='plotly_white')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.line(data, x='Year', y='Inflation', 
                  title=t['inflation_rate'], markers=True,
                  labels={'Inflation': t['inflation_rate'], 'Year': 'السنة'})
    fig2.update_layout(template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)

# قسم التنبؤات
st.subheader(t['forecasts'])
forecast_col1, forecast_col2 = st.columns(2)

# الحصول على التنبؤات
gdp_years, gdp_forecast = forecast_gdp('ARIMA')
inflation_years, inflation_forecast = forecast_inflation('ARIMA')

with forecast_col1:
    for i, year in enumerate(gdp_years):
        change = gdp_forecast[i] - data[data['Year'] == data['Year'].max()]['GDP Growth'].values[0]
        arrow = "▲" if change > 0 else "▼"
        color = "#27ae60" if change > 0 else "#e74c3c"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{t['gdp_forecast']} ({year})</div>
            <div class="metric-value">{gdp_forecast[i]:.1f}%</div>
            <div style="font-size:12px;color:{color};">
                {arrow} {abs(change):.1f}% عن آخر قياس
            </div>
        </div>
        """, unsafe_allow_html=True)

with forecast_col2:
    for i, year in enumerate(inflation_years):
        change = inflation_forecast[i] - data[data['Year'] == data['Year'].max()]['Inflation'].values[0]
        arrow = "▲" if change > 0 else "▼"
        color = "#e74c3c" if change > 0 else "#27ae60"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{t['inflation_forecast']} ({year})</div>
            <div class="metric-value">{inflation_forecast[i]:.1f}%</div>
            <div style="font-size:12px;color:{color};">
                {arrow} {abs(change):.1f}% عن آخر قياس
            </div>
        </div>
        """, unsafe_allow_html=True)

# مقارنة النماذج
st.subheader(t['model_comparison'])
model_options = ['ARIMA', 'Random Forest']

# إنشاء بيانات المقارنة
comparison_data = []
for model in model_options:
    _, gdp_fc = forecast_gdp(model)
    _, inf_fc = forecast_inflation(model)
    comparison_data.append({
        'Model': model,
        'GDP Forecast': gdp_fc[0],
        'Inflation Forecast': inf_fc[0]
    })

model_data = pd.DataFrame(comparison_data)

# رسم بياني مقارنة
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

sns.barplot(x='Model', y='GDP Forecast', data=model_data, ax=ax1, palette='Blues_d')
ax1.set_title(t['gdp_forecast'], fontsize=14)
ax1.set_ylabel('%')

sns.barplot(x='Model', y='Inflation Forecast', data=model_data, ax=ax2, palette='Reds_d')
ax2.set_title(t['inflation_forecast'], fontsize=14)
ax2.set_ylabel('%')

plt.tight_layout()
st.pyplot(fig)

# تقييم أداء النماذج
st.subheader(t['model_performance'])

# حساب أداء النماذج
def calculate_performance(data, target):
    train_data = data.iloc[:-5]  # تدريب على جميع البيانات ما عدا آخر 5 سنوات
    test_data = data.iloc[-5:]   # اختبار على آخر 5 سنوات
    
    # ARIMA
    arima_model = ARIMA(train_data[target], order=(2,1,2)).fit()
    arima_pred = arima_model.forecast(steps=len(test_data))
    arima_rmse = np.sqrt(mean_squared_error(test_data[target], arima_pred))
    
    # Random Forest
    X_train = train_data[['Year']]
    y_train = train_data[target]
    X_test = test_data[['Year']]
    y_test = test_data[target]
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    
    return {
        'ARIMA': arima_rmse,
        'Random Forest': rf_rmse
    }

gdp_performance = calculate_performance(data, 'GDP Growth')
inflation_performance = calculate_performance(data, 'Inflation')

# عرض أداء النماذج
perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.markdown("**أداء نماذج الناتج المحلي (RMSE)**")
    st.dataframe(pd.DataFrame.from_dict(gdp_performance, 
                                       orient='index',
                                       columns=['القيمة']).rename_axis('النموذج').reset_index(),
                hide_index=True)

with perf_col2:
    st.markdown("**أداء نماذج التضخم (RMSE)**")
    st.dataframe(pd.DataFrame.from_dict(inflation_performance, 
                                       orient='index',
                                       columns=['القيمة']).rename_axis('النموذج').reset_index(),
                hide_index=True)

# أحدث البيانات
st.subheader(t['latest_data'])
latest_year = data['Year'].max()
latest_data = data[data['Year'] == latest_year].set_index('Year').T
st.dataframe(latest_data.style.format("{:.1f}"), use_container_width=True)

# تحليل متقدم
st.sidebar.header("⚙️ " + t['select_model'])
model_options = ['ARIMA', 'Random Forest']
selected_model = st.sidebar.selectbox(t['select_model'], model_options)

indicator_options = [t['gdp_growth'], t['inflation_rate']]
selected_indicator = st.sidebar.selectbox(t['select_indicator'], indicator_options)

# خيارات متقدمة
st.sidebar.header("⚡ خيارات متقدمة")
forecast_years = st.sidebar.slider("سنوات التنبؤ", 1, 5, 2)
confidence_level = st.sidebar.slider("مستوى الثقة (%)", 80, 99, 95)

if st.sidebar.button("🚀 " + t['run_analysis']):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"{t['analysis_in_progress']} {i+1}%")
        time.sleep(0.02)
    
    status_text.success(f"✅ {t['forecasts']} {t['completed_successfully']}")
    
    # عرض النتائج
    if selected_indicator == t['gdp_growth']:
        years, forecast = forecast_gdp(selected_model, forecast_years)
        current_value = data[data['Year'] == data['Year'].max()]['GDP Growth'].values[0]
        
        st.subheader(f"{t['gdp_forecast']} - {selected_model}")
        
        # رسم التنبؤ
        forecast_df = pd.DataFrame({
            'Year': years,
            'Forecast': forecast,
            'Actual': [None] * len(years)
        })
        
        # إنشاء مخطط شامل
        full_data = data[['Year', 'GDP Growth']].rename(columns={'GDP Growth': 'Actual'})
        full_data['Type'] = 'تاريخي'
        forecast_df['Type'] = 'تنبؤ'
        combined = pd.concat([full_data, forecast_df[['Year', 'Forecast', 'Type']].rename(columns={'Forecast': 'Actual'})])
        
        fig = px.line(combined, x='Year', y='Actual', color='Type', 
                     markers=True, title=t['gdp_forecast'],
                     labels={'Actual': t['gdp_growth'], 'Year': 'السنة'})
        st.plotly_chart(fig, use_container_width=True)
        
        # عرض القيم
        for i, year in enumerate(years):
            st.metric(f"{t['gdp_forecast']} {year}", 
                     f"{forecast[i]:.1f}%",
                     delta=f"{forecast[i] - current_value:.1f}%")
            
    else:
        years, forecast = forecast_inflation(selected_model, forecast_years)
        current_value = data[data['Year'] == data['Year'].max()]['Inflation'].values[0]
        
        st.subheader(f"{t['inflation_forecast']} - {selected_model}")
        
        # رسم التنبؤ
        forecast_df = pd.DataFrame({
            'Year': years,
            'Forecast': forecast,
            'Actual': [None] * len(years)
        })
        
        # إنشاء مخطط شامل
        full_data = data[['Year', 'Inflation']].rename(columns={'Inflation': 'Actual'})
        full_data['Type'] = 'تاريخي'
        forecast_df['Type'] = 'تنبؤ'
        combined = pd.concat([full_data, forecast_df[['Year', 'Forecast', 'Type']].rename(columns={'Forecast': 'Actual'})])
        
        fig = px.line(combined, x='Year', y='Actual', color='Type', 
                     markers=True, title=t['inflation_forecast'],
                     labels={'Actual': t['inflation_rate'], 'Year': 'السنة'})
        st.plotly_chart(fig, use_container_width=True)
        
        # عرض القيم
        for i, year in enumerate(years):
            st.metric(f"{t['inflation_forecast']} {year}", 
                     f"{forecast[i]:.1f}%",
                     delta=f"{forecast[i] - current_value:.1f}%")

# قسم التقارير
st.sidebar.header("📊 " + t['download_report'])
report_type = st.sidebar.radio("نوع التقرير", ["تقرير مختصر", "تقرير مفصل"])

if st.sidebar.button("📥 " + t['download_report']):
    with st.spinner("جاري إنشاء التقرير..."):
        progress_bar = st.sidebar.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"EconoPredict_Report_{timestamp}.pdf"
        
        # محاكاة إنشاء ملف PDF
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font('Noto', '', 'NotoNaskhArabic-Regular.ttf', uni=True)
        pdf.set_font('Noto', size=14)
        
        pdf.cell(200, 10, txt=f"تقرير EconoPredict - {datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
        pdf.cell(200, 10, txt=f"نموذج: {selected_model}", ln=True)
        pdf.cell(200, 10, txt=f"المؤشر: {selected_indicator}", ln=True)
        pdf.cell(200, 10, txt="", ln=True)
        
        if selected_indicator == t['gdp_growth']:
            years, forecast = forecast_gdp(selected_model, forecast_years)
            for i, year in enumerate(years):
                pdf.cell(200, 10, txt=f"تنبؤ الناتج المحلي لسنة {year}: {forecast[i]:.1f}%", ln=True)
        else:
            years, forecast = forecast_inflation(selected_model, forecast_years)
            for i, year in enumerate(years):
                pdf.cell(200, 10, txt=f"تنبؤ التضخم لسنة {year}: {forecast[i]:.1f}%", ln=True)
                
        pdf.output(filename)
        
    st.sidebar.success(f"✅ {t['report_generated']}")
    with open(filename, "rb") as file:
        st.sidebar.download_button(
            label="⬇️ " + t['download_report'],
            data=file,
            file_name=filename,
            mime="application/pdf"
        )
    os.remove(filename)

# تذييل الصفحة
st.markdown("---")
st.markdown("""
<div class="footer">
    <strong>EconoPredict</strong> - نظام متقدم للتنبؤ الاقتصادي<br>
    تم تطويره بواسطة: يوسف اولاد محمد<br>
    © 2023 جميع الحقوق محفوظة | الإصدار 2.1
</div>
""", unsafe_allow_html=True)
