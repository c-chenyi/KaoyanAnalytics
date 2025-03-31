import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
import pyecharts.options as opts
from pyecharts.charts import Bar, Line, Pie, Map, Tab
import jieba
import matplotlib.pyplot as plt
from matplotlib.image import imread
import stylecloud
import os
import webbrowser
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# 设置全局参数
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "xxxxxx",
    "database": "kaoyan"
}


# 初始化数据库连接
def init_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"数据库连接错误: {e}")
        return None


# 创建数据表
def create_tables(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS 考研分数线数据 (
            年份 INT,
            学校名称_链接 VARCHAR(255),
            学校名称 VARCHAR(255),
            院系名称_链接 VARCHAR(255),
            院系名称 VARCHAR(255),
            专业代码 VARCHAR(20),
            专业名称_链接 VARCHAR(255),
            专业名称 VARCHAR(255),
            总分 VARCHAR(10),
            政治__管综 VARCHAR(10),
            外语 VARCHAR(10),
            业务课_一 VARCHAR(10),
            业务课_二 VARCHAR(10)
        )""")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS 调剂信息 (
            school VARCHAR(255),
            name VARCHAR(255),
            time VARCHAR(50),
            province VARCHAR(50),
            school_level VARCHAR(20),
            school_types VARCHAR(20)
        )""")
        conn.commit()
    except Error as e:
        print(f"创建表时出错: {e}")
        conn.rollback()
    finally:
        cursor.close()


# 批量插入数据
def batch_insert(cursor, query, data, batch_size=1000):
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        cursor.executemany(query, batch)
    return len(data)


# 处理考研分数线数据
def process_cutoff_data(conn):
    cursor = conn.cursor()
    csv_files = [f'./考研历年国家分数线({i}).csv' for i in range(1, 7)]
    csv_files = [f for f in csv_files if os.path.exists(f)]
    if not csv_files:
        print("未找到CSV文件")
        return None
    insert_query = """
    INSERT INTO 考研分数线数据 (年份, 学校名称_链接, 学校名称, 院系名称_链接, 院系名称, 专业代码, 专业名称_链接, 专业名称, 总分, 政治__管综, 外语, 业务课_一, 业务课_二)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    all_data = []
    for file in tqdm(csv_files, desc="处理CSV文件"):
        try:
            df = pd.read_csv(file)
            df = df.drop_duplicates().dropna()
            all_data.extend([
                (row['年份'], row['学校名称_链接'], row['学校名称'], row['院系名称_链接'], row['院系名称'],
                 row['专业代码'], row['专业名称_链接'], row['专业名称'], row['总分'],
                 row['政治__管综'], row['外语'], row['业务课_一'], row['业务课_二']
                 ) for _, row in df.iterrows()
            ])
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
    if all_data:
        total = batch_insert(cursor, insert_query, all_data)
        print(f"成功插入 {total} 条分数线数据")
        conn.commit()
    cursor.execute("SELECT * FROM 考研分数线数据")
    columns = [col[0] for col in cursor.description]
    data = cursor.fetchall()
    df_all = pd.DataFrame(data, columns=columns)
    df_all = df_all.drop_duplicates().dropna()
    df_all = df_all.drop(['学校名称_链接', '院系名称_链接', '专业名称_链接'], axis=1)
    df_all['专业名称'] = df_all['专业名称'].str.replace(r'\(专业学位\)|★', '', regex=True)
    cursor.close()
    return df_all


# 处理调剂信息数据
def process_adjustment_data(conn):
    cursor = conn.cursor()
    try:
        df_info = pd.read_excel('./大学信息2021new.xlsx')
        df = pd.read_excel('./考研调剂数据-3.08.xlsx')
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return None

    def transform_attr(x):
        if pd.isna(x): return '其他'
        x = str(x)
        if '211' in x and '985' not in x:
            return '211'
        elif '985' in x:
            return '985'
        return '双非'

    def transform_type(x):
        if pd.isna(x): return '其他'
        x = str(x)
        if any(t in x for t in ['理工类', '理工科', '理工']):
            return '理工'
        elif any(t in x for t in ['综合类', '综合性', '综合']):
            return '综合'
        elif '师范' in x:
            return '师范'
        elif '农林' in x or '农业' in x:
            return '农林'
        elif '医药' in x:
            return '医药'
        elif '民族' in x:
            return '民族'
        return '其他'

    df_info['school_level'] = df_info['school_attr'].astype(str).apply(transform_attr)
    df_info['school_types'] = df_info['school_type'].astype(str).apply(transform_type)
    df_info = df_info[['school', 'province', 'school_level', 'school_types']]
    province_mapping = {
        '北京工商大学': '北京', '哈尔滨工程大学': '黑龙江', '江苏大学': '江苏',
        '青岛大学': '山东', '北京石油化工学院': '北京', '齐鲁工业大学': '山东',
        '江苏科技大学': '江苏', '浙江农林大学': '浙江', '燕山大学': '河北',
        '福州大学': '福建', '内蒙古科技大学': '内蒙古'
    }
    df_info['province'] = df_info.apply(lambda x: province_mapping.get(x['school'], x['province']), axis=1)
    df_info = df_info.drop_duplicates()
    df_2021 = df[df['time'].str.contains('2021', na=False)].copy()
    df_all_info = pd.merge(df_2021, df_info, how='left', on='school')
    df_all_info = df_all_info[['school', 'name', 'time', 'province', 'school_level', 'school_types']]
    insert_query = """
    INSERT INTO 调剂信息 (school, name, time, province, school_level, school_types)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    data = [(row['school'], row['name'], row['time'], row['province'],
             row['school_level'], row['school_types']) for _, row in df_all_info.iterrows()]
    if data:
        total = batch_insert(cursor, insert_query, data)
        print(f"成功插入 {total} 条调剂信息数据")
        conn.commit()
    cursor.close()
    return df_all_info


# 生成词云
def generate_wordcloud(text_series, output_path, icon_name='fas fa-book'):
    def get_cut_words(content_series):
        stop_words = [' ', '是', '的', '查看', '详细', '详见', '详情', '与化', '03', '02', '01', '正文', '多个', '相关']
        content = content_series.str.cat(sep='。')
        words = jieba.lcut(content)
        return [word for word in words if word not in stop_words and len(word) >= 2]

    text = ' '.join(get_cut_words(text_series))
    stylecloud.gen_stylecloud(
        text=text, collocations=False,
        font_path='./SimHei.ttf' if os.path.exists('./SimHei.ttf') else None,
        icon_name=icon_name, size=768, output_name=output_path
    )
    if os.path.exists(output_path):
        img = imread(output_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print(f"词云生成失败: {output_path} 不存在")


# 优化交互性可视化
def visualize_data_interactive(df_all, df_all_info):
    tab = Tab()
    data_2020 = df_all[df_all['年份'] == 2020].copy()

    # 处理分数数据，确保数值有效
    def transform_score(x):
        try:
            if '-' in str(x) or pd.isna(x):
                return None  # 使用 None 以便后续过滤
            return int(float(x))
        except (ValueError, TypeError):
            return None

    # 柱状图：各专业分数统计
    data_2020['总分'] = data_2020['总分'].apply(transform_score)
    # 过滤掉无效分数并按专业分组计算统计值
    data_2020_filtered = data_2020.dropna(subset=['总分'])
    if not data_2020_filtered.empty:
        data_1 = (data_2020_filtered.groupby('专业名称')['总分']
                  .agg(['mean', 'max', 'min'])
                  .dropna()
                  .sort_values('mean', ascending=False)[:50])
        data_1['mean'] = data_1['mean'].astype(int)
        data_1.columns = ['mean', 'amax', 'amin']

        # 检查是否有数据异常（最高、最低、平均值相同）
        problematic_rows = data_1[data_1['amax'] == data_1['amin']].index
        if len(problematic_rows) > 0:
            print(f"警告：以下专业最高分、最低分、平均分相同，可能数据有误：{problematic_rows.tolist()}")

        if not data_1.empty:
            bar = (
                Bar(init_opts=opts.InitOpts(theme='light', width='1400px', height='800px'))
                .add_xaxis(data_1.index.tolist())
                .add_yaxis('最高分', data_1['amax'].tolist(),
                           category_gap="30%",
                           label_opts=opts.LabelOpts(is_show=False),
                           itemstyle_opts=opts.ItemStyleOpts(color='#ec9bad'))
                .add_yaxis('最低分', data_1['amin'].tolist(),
                           category_gap="30%",
                           label_opts=opts.LabelOpts(is_show=False),
                           itemstyle_opts=opts.ItemStyleOpts(color='#87CEFA'))
                .add_yaxis('平均分', data_1['mean'].tolist(),
                           category_gap="30%",
                           label_opts=opts.LabelOpts(is_show=False),
                           itemstyle_opts=opts.ItemStyleOpts(color='#90EE90'))
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="各专业分数统计（2020）"),
                    xaxis_opts=opts.AxisOpts(
                        axislabel_opts=opts.LabelOpts(rotate=45, font_size=12, margin=10),
                        name="专业名称",
                        name_location="middle",
                        name_gap=40
                    ),
                    yaxis_opts=opts.AxisOpts(
                        name="分数",
                        min_=300,  # 设置Y轴起点为300分
                        max_=data_1['amax'].max() + 10,  # 动态设置最大值
                        interval=10,  # 设置Y轴刻度间隔为10分
                        axislabel_opts=opts.LabelOpts(formatter="{value} 分")  # 显示单位
                    ),
                    datazoom_opts=[
                        opts.DataZoomOpts(type_='slider', range_start=0, range_end=100),
                        opts.DataZoomOpts(type_='inside')
                    ],
                    tooltip_opts=opts.TooltipOpts(trigger="axis", formatter="{a}: {c} 分"),
                    toolbox_opts=opts.ToolboxOpts(is_show=True, feature={
                        "dataView": {"show": True},
                        "restore": {"show": True},
                        "saveAsImage": {"show": True}
                    }),
                    legend_opts=opts.LegendOpts(type_="scroll", pos_left="right", orient="vertical")
                )
            )
            tab.add(bar, "专业分数统计")
        else:
            print("2020年专业分数数据为空，无法生成柱状图")
    else:
        print("2020年数据过滤后为空，可能分数数据无效")

    # 如果调剂信息为空，提前返回
    if df_all_info is None or df_all_info.empty:
        print("无调剂信息数据可分析")
        report_file = "interactive_analysis_report.html"
        tab.render(report_file)
        webbrowser.open_new_tab(f"file://{os.path.abspath(report_file)}")
        return

    # 折线图：发布时间趋势
    pub_time = df_all_info['time'].value_counts().sort_index()
    if not pub_time.empty:
        line_chart = (
            Line(init_opts=opts.InitOpts(width="1200px", height="600px"))
            .add_xaxis(pub_time.index.tolist())
            .add_yaxis("发布数量", pub_time.values.tolist(), is_smooth=True,
                       markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max"),
                                                              opts.MarkPointItem(type_="min")]),
                       markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="average")]),
                       areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
                       label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                title_opts=opts.TitleOpts(title="调剂信息发布时间趋势"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
                yaxis_opts=opts.AxisOpts(name="数量"),
                datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                toolbox_opts=opts.ToolboxOpts(is_show=True),
                visualmap_opts=opts.VisualMapOpts(is_show=True, min_=0, max_=pub_time.max())
            )
        )
        tab.add(line_chart, "发布时间趋势")
    else:
        print("调剂信息发布时间数据为空")

    # 饼图：学校层次分布
    level_data = df_all_info['school_level'].value_counts()
    if not level_data.empty:
        pie_chart = (
            Pie(init_opts=opts.InitOpts(width="800px", height="600px"))
            .add(series_name="学校层次",
                 data_pair=[list(z) for z in zip(level_data.index.tolist(), level_data.tolist())],
                 radius=["40%", "75%"],
                 label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)", position="outside"),
                 rosetype="radius")
            .set_global_opts(
                title_opts=opts.TitleOpts(title="学校层次分布"),
                legend_opts=opts.LegendOpts(orient="vertical", pos_left="left"),
                tooltip_opts=opts.TooltipOpts(trigger="item"),
                toolbox_opts=opts.ToolboxOpts(is_show=True)
            )
        )
        tab.add(pie_chart, "学校层次分布")
    else:
        print("学校层次数据为空")

    # 地图：各省调剂信息分布
    if 'province' in df_all_info.columns:
        province_dist = df_all_info['province'].value_counts().dropna()
        if not province_dist.empty:
            province_dist.index = province_dist.index.str.replace(r'省|市|自治区|特别行政区', '', regex=True)
            map_chart = (
                Map()
                .add("调剂信息数量",
                     [list(z) for z in zip(province_dist.index.tolist(), province_dist.tolist())],
                     "china",
                     is_map_symbol_show=True)
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="各省调剂信息分布"),
                    visualmap_opts=opts.VisualMapOpts(max_=province_dist.max(), is_piecewise=True),
                    tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c}"),
                    toolbox_opts=opts.ToolboxOpts(is_show=True)
                )
                .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            )
            tab.add(map_chart, "省份分布")
        else:
            print("省份分布数据为空")
    else:
        print("调剂信息中缺少省份列")

    # 渲染并打开文件
    report_file = "interactive_analysis_report.html"
    tab.render(report_file)
    webbrowser.open_new_tab(f"file://{os.path.abspath(report_file)}")
    print(f"交互式报告已生成: {os.path.abspath(report_file)}")



# 预测分数线趋势（线性回归 + ARIMA，多步预测，模型评估）
def predict_score_trend(df_all):
    df_scores = df_all[['年份', '专业名称', '总分']].copy()
    df_scores['总分'] = df_scores['总分'].apply(lambda x: 0 if '-' in str(x) else int(float(x)))
    df_scores = df_scores[df_scores['总分'] > 0]
    df_agg = df_scores.groupby(['年份', '专业名称'])['总分'].mean().reset_index()
    prof_counts = df_agg['专业名称'].value_counts()
    valid_profs = prof_counts[prof_counts >= 3].index
    df_agg = df_agg[df_agg['专业名称'].isin(valid_profs)]

    predictions_lr = {}  # 线性回归预测
    predictions_arima = {}  # ARIMA 预测
    actual_data = {}
    metrics = {}  # 存储评估指标

    future_years = np.array([[2021], [2022], [2023]])  # 多步预测

    for prof in valid_profs[:5]:  # 前5个专业
        prof_data = df_agg[df_agg['专业名称'] == prof]
        X = prof_data['年份'].values.reshape(-1, 1)
        y = prof_data['总分'].values

        # 线性回归
        model_lr = LinearRegression()
        model_lr.fit(X, y)
        pred_scores_lr = model_lr.predict(future_years)
        pred_train_lr = model_lr.predict(X)  # 训练集预测用于评估

        # ARIMA
        model_arima = ARIMA(y, order=(1, 1, 0))
        model_fit = model_arima.fit()
        pred_scores_arima = model_fit.forecast(steps=3)  # 预测3年

        # 评估指标
        r2_lr = r2_score(y, pred_train_lr)  # 线性回归 R²
        rmse_lr = np.sqrt(mean_squared_error(y, pred_train_lr))  # 线性回归 RMSE
        # ARIMA 的 RMSE 使用历史数据的拟合值与实际值比较
        arima_fitted = model_fit.fittedvalues
        rmse_arima = np.sqrt(mean_squared_error(y[1:], arima_fitted[1:]))  # 去掉首项（差分后少一项）

        metrics[prof] = {
            'Linear Regression R²': r2_lr,
            'Linear Regression RMSE': rmse_lr,
            'ARIMA RMSE': rmse_arima
        }

        # 存储结果
        years = prof_data['年份'].tolist() + [2021, 2022, 2023]
        actual_data[prof] = prof_data['总分'].tolist() + [None] * 3
        predictions_lr[prof] = prof_data['总分'].tolist() + pred_scores_lr.tolist()
        predictions_arima[prof] = prof_data['总分'].tolist() + pred_scores_arima.tolist()

    # 打印评估指标
    for prof, metric in metrics.items():
        print(f"\n专业: {prof}")
        print(f"线性回归 R²: {metric['Linear Regression R²']:.4f}")
        print(f"线性回归 RMSE: {metric['Linear Regression RMSE']:.2f}")
        print(f"ARIMA RMSE: {metric['ARIMA RMSE']:.2f}")

    # 可视化
    line_chart = Line(init_opts=opts.InitOpts(width="1200px", height="600px"))
    for prof in predictions_lr.keys():
        line_chart.add_xaxis(years)
        line_chart.add_yaxis(
            f"{prof} (实际)", actual_data[prof],
            label_opts=opts.LabelOpts(is_show=False), linestyle_opts=opts.LineStyleOpts(width=2)
        )
        line_chart.add_yaxis(
            f"{prof} (线性回归)", predictions_lr[prof],
            label_opts=opts.LabelOpts(is_show=False), linestyle_opts=opts.LineStyleOpts(width=2, type_="dashed")
        )
        line_chart.add_yaxis(
            f"{prof} (ARIMA)", predictions_arima[prof],
            label_opts=opts.LabelOpts(is_show=False), linestyle_opts=opts.LineStyleOpts(width=2, type_="dotted")
        )
    line_chart.set_global_opts(
        title_opts=opts.TitleOpts(title="专业分数线趋势预测（至2023年）"),
        datazoom_opts=[opts.DataZoomOpts()],
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        legend_opts=opts.LegendOpts(pos_left="right", orient="vertical")
    )
    output_file = "score_trend_prediction.html"
    line_chart.render(output_file)
    webbrowser.open_new_tab(f"file://{os.path.abspath(output_file)}")
    print(f"分数线预测图表已生成: {os.path.abspath(output_file)}")


# 主函数
def main():
    conn = init_db_connection()
    if conn is None:
        return
    try:
        create_tables(conn)
        df_all = process_cutoff_data(conn)
        df_all_info = process_adjustment_data(conn)
        if df_all is not None:
            generate_wordcloud(df_all['专业名称'], './专业词云.png', 'fas fa-book')
            generate_wordcloud(df_all['学校名称'], './学校词云.png', 'fas fa-graduation-cap')
        if df_all is not None and df_all_info is not None:
            visualize_data_interactive(df_all, df_all_info)
            predict_score_trend(df_all)
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        conn.close()
        print("数据库连接已关闭")


if __name__ == "__main__":
    main()
