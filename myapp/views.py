import base64
import pickle
from django.shortcuts import render, redirect
from django.http import HttpResponse
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO
from .forms import CsvUploadForm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


def base(request):
    return render(request, 'index.html')

def fileupload(request):
    if request.method == 'POST':
        form = CsvUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                form.save()
                file = form.cleaned_data['file']
            
                df = pd.read_csv(file, delimiter=',', encoding='utf-8', skiprows=0, header=0)
                df.dropna(axis=0,inplace=True)

                df_base64 = base64.b64encode(pickle.dumps(df)).decode('utf-8')
                request.session['df'] = df_base64
                return redirect('choose_algo')
            except Exception as e:

                return HttpResponse(f'Error: {str(e)}')
    else:
        form = CsvUploadForm()

    return render(request, 'file_upload.html', {'form': form})

def column_select(request, algorithm):
    df_base64 = request.session.get('df', '') 
    df = pickle.loads(base64.b64decode(df_base64))

    columns = df.columns
    if request.method == 'POST':
        print(request.POST)
        selected_columns_x = request.POST.getlist('columns_x')
        selected_columns_y = request.POST.getlist('columns_y')
        k_neighbors = int(request.POST.get('k_neighbors', 5))
        print(selected_columns_x)
        print(selected_columns_y)
        print("in form")
        if algorithm == 'regression':
            print("in reg")
            request.session['selected_columns_x'] = selected_columns_x
            request.session['selected_columns_y'] = selected_columns_y
            return redirect('compute_regression_algorithm')
        elif algorithm == 'knn':
            print("in knn")
            request.session['selected_columns_x'] = selected_columns_x
            request.session['selected_columns_y'] = selected_columns_y
            return redirect('compute_knn_algorithm', algorithm=algorithm, k_neighbors=k_neighbors)
        elif algorithm == 'data_visual':
            print("in datavisual")
            request.session['selected_columns_x'] = selected_columns_x
            request.session['selected_columns_y'] = selected_columns_y
            return redirect('predictions')


    context = {
        'columns': columns,
        'algorithm': algorithm,
    }

    return render(request, 'selection.html', context)

def compute_knn_algorithm(request, algorithm, k_neighbors=None):
    df_base64 = request.session.get('df', '')
    df = pickle.loads(base64.b64decode(df_base64))

    selected_columns_x = request.session.get('selected_columns_x', None)
    selected_columns_y = request.session.get('selected_columns_y', None)

    x = df[selected_columns_x]
    y = df[selected_columns_y]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101, shuffle=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    k_neighbors = int(request.POST.get('k_neighbors', k_neighbors or 5))

    model = KNeighborsRegressor(n_neighbors=k_neighbors)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mean_sq = mean_squared_error(y_test, y_pred)

    fig_knn, ax1 = plt.subplots(figsize=(12, 6))
    plt.scatter(X_test[selected_columns_x[0]], y_test, color='blue', label='Actual')
    plt.scatter(X_test[selected_columns_x[0]], y_pred, color='red', label='Predicted')
    
    plt.xlabel(selected_columns_x[0])
    plt.ylabel(selected_columns_y[0])
    plt.legend()
    plt.title(f"KNN Regression Analysis (k_neighbors={k_neighbors})")

    buf_knn = BytesIO()
    canvas_knn = FigureCanvasAgg(fig_knn)
    canvas_knn.print_png(buf_knn)
    image_base64_knn = base64.b64encode(buf_knn.getvalue()).decode('utf-8')
    plt.close(fig_knn)

    context = {
        'y_pred': y_pred,
        'mean_sq': mean_sq,
        'image_base64_knn': image_base64_knn,
        'k_neighbors': k_neighbors, 
    }

    return render(request, 'knn_an.html', context)

def compute_regression_algorithm(request):
    df_base64 = request.session.get('df', '') 
    df = pickle.loads(base64.b64decode(df_base64))

    selected_columns_x = request.session.get('selected_columns_x', None)
    selected_columns_y = request.session.get('selected_columns_y', None)

    x = df[selected_columns_x]
    y = df[selected_columns_y]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101, shuffle=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression().fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mean_sq = mean_squared_error(y_test, y_pred)

    r_squared = model.score(x,y)

    intercept = model.intercept_

    fig_reg, ax1 = plt.subplots(figsize=(12, 6))
    plt.scatter(X_test[selected_columns_x[0]], y_test, color='blue', label='Actual')
    plt.scatter(X_test[selected_columns_x[0]], y_pred, color='red', label='Predicted')
    
    plt.xlabel(selected_columns_x[0])
    plt.ylabel(selected_columns_y[0])
    plt.legend()
    plt.title("Regression Analysis")

    buf_reg = BytesIO()
    canvas_reg = FigureCanvasAgg(fig_reg)
    canvas_reg.print_png(buf_reg)
    image_base64_reg = base64.b64encode(buf_reg.getvalue()).decode('utf-8')
    plt.close(fig_reg)

    context = {
        'y_pred': y_pred,
        'mean_sq': mean_sq,
        'r_squared' : r_squared,
        'intercept' : intercept,
        'image_base64_reg': image_base64_reg,
    }

    return render(request, 'regression_an.html', context)


def predictions(request):
    df_base64 = request.session.get('df', '') 
    df = pickle.loads(base64.b64decode(df_base64))

    columns = df.columns
    count = df.shape[1]

    selected_columns_x = request.session.get('selected_columns_x', None)
    selected_columns_y = request.session.get('selected_columns_y', None)

    x = selected_columns_x
    y = selected_columns_y



    # Boxplot
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x=selected_columns_x[0], y=selected_columns_y[0], ax=ax1)
    plt.title(f"Box Plot of {selected_columns_x[0]} vs {selected_columns_y[0]}")
    plt.xlabel(selected_columns_x[0])
    plt.ylabel(selected_columns_y[0])
    plt.xticks(rotation=90)

    buf1 = BytesIO()
    canvas1 = FigureCanvasAgg(fig1)
    canvas1.print_png(buf1)
    image_base64_1 = base64.b64encode(buf1.getvalue()).decode('utf-8')
    plt.close(fig1)

    # Bar Chart of x
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    df[selected_columns_x[0]].value_counts().plot(kind='bar', ax=ax2)
    plt.title(f"{selected_columns_x[0]} counts representation")
    plt.xlabel(selected_columns_x[0])
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    buf2 = BytesIO()
    canvas2 = FigureCanvasAgg(fig2)
    canvas2.print_png(buf2)
    image_base64_2 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    plt.close(fig2)

    # Bar chart of y
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    df[selected_columns_y[0]].value_counts().plot(kind='bar', ax=ax3)
    plt.title(f"Bar Chart of : {selected_columns_y[0]}")
    plt.xlabel(selected_columns_x[0])
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    buf3 = BytesIO()
    canvas3 = FigureCanvasAgg(fig3)
    canvas3.print_png(buf3)
    image_base64_3 = base64.b64encode(buf3.getvalue()).decode('utf-8')
    plt.close(fig3)

    # Scatter plot
    er = None
    image_base64_4 = None
    if selected_columns_x[0] in df.columns and selected_columns_y[0] in df.columns:
        x_col = df[selected_columns_x[0]].astype(float)
        y_col = df[selected_columns_y[0]].astype(float)

        if len(x_col) == len(y_col):
            fig4, ax4 = plt.subplots(figsize=(12,6))
            plt.scatter(x_col, y_col)
            plt.title(f"Scatter Plot of {selected_columns_x[0]} vs {selected_columns_y[0]}")
            plt.xlabel(selected_columns_x[0])
            plt.ylabel(selected_columns_y[0])

            buf4 = BytesIO()
            canvas4 = FigureCanvasAgg(fig4)
            canvas4.print_png(buf4)
            image_base64_4 = base64.b64encode(buf4.getvalue()).decode('utf-8')
            plt.close(fig4)
        else:
            er = "Scatter Plot can't be generated as the columns are not of the same length"
            print("Error: Not of the same length")
            x_col = []
            y_col = []

    # Line plot
    fig5, ax5 = plt.subplots(figsize=(12,6))
    sns.lineplot(data=df, x=selected_columns_x[0], y=selected_columns_y[0], ax=ax5)
    plt.title(f"Line plot of : {selected_columns_x[0]} vs {selected_columns_y[0]}")
    plt.xlabel(selected_columns_x[0])
    plt.ylabel(selected_columns_y[0])

    buf5 = BytesIO()
    canvas5 = FigureCanvasAgg(fig5)
    canvas5.print_png(buf5)
    image_base64_5 = base64.b64encode(buf5.getvalue()).decode('utf-8')
    plt.close(fig5)

    # Pie chart
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    col = df[selected_columns_x[0]].value_counts()
    plt.pie(col, labels=col.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title(f"{selected_columns_x[0]} counts representation in Pie Chart")
    buf6 = BytesIO()
    canvas6 = FigureCanvasAgg(fig6)
    canvas6.print_png(buf6)

    image_base64_6 = base64.b64encode(buf6.getvalue()).decode('utf-8')
    plt.close(fig6)



    context = {
        'image_base64_1': image_base64_1, 
        'image_base64_2': image_base64_2,
        'image_base64_3': image_base64_3,
        'image_base64_4': image_base64_4,
        'image_base64_5': image_base64_5,
        'image_base64_6': image_base64_6,
        'columns' : columns,
        'er' : er,
        'x' : x,
        'y' : y,

    }

    return render(request, 'predictions.html', context)



def choose_algo(request):
    return render(request, "choose_algo.html")