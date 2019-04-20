from bs4 import BeautifulSoup
from pathlib import Path
from requests import get
from multiprocessing import Pool
from sklearn import metrics, linear_model
from tornado import template

import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import math
import datetime
import time
import pygal as pygal
import pdfkit

def multiplesof3():
        n = 3
        i = 3
        while n < 20:
            print(n)
            n += i

def is_pal(s,t):
    if len(t) <= 1:
        print (s + " is palindrome")
        return 
    if t[0] != t[-1]:
        return 
    return is_pal(s,t[1:-1])

def analyze_line(l):
    words = l.split(' ')
    num_words = len(words)
    num_chars = len(''.join(words))
    return num_chars, num_words

def analyze_file(path):
    chars, words, lines = 0, 0, 0
    try:
        with open(path) as f:
            for line in f:
                c, w = analyze_line(line)
                chars += c
                words += w
                lines += 1
        print('lines: ' + str(lines) + ', words: ' + str(words) + ', chars: ' + str(chars))
    except FileNotFoundError:
        print('File not found: \'' + path + '\'')

def analyze_data():
    salaries = pd.read_csv('data\\salaries.csv', header=0)
    lawyers = salaries.loc[salaries['Job'] == 'Lawyers']
    lawyers.sort_values(by=['Salary'], ascending=False, inplace=True)
    lawyers.reset_index(inplace=True, drop=True)
    print("cities sorted by descending order of lawyer salary")
    print(lawyers)
    salaries.drop(['City'], axis=1, inplace=True)
    print("\n Median salary of each profession")
    print(salaries.groupby(['Job']).median())


def scrape_imdb():
        # Create URL
        base_url = 'https://www.imdb.com'
        url = base_url + '/search/title?title_type=feature&countries=us&languages=en&count=250'
        contains_next = True
        n = 0
        while contains_next:
            ans = []
            response = get(url)
            html_soup = BeautifulSoup(response.text, 'html.parser')
            temp_c = html_soup.find_all('div', class_='lister-item mode-advanced')
            movie_containers = temp_c
            for mc in movie_containers:
                try:
                    title = mc.h3.a.text
                    rating = float(mc.strong.text)
                    votes_str = mc.find('span', attrs={'name': 'nv'}).text
                    num_votes = int(votes_str.replace(',', ''))
                    y_str = mc.h3.find('span', class_='lister-item-year text-muted unbold').text[1:5]
                    year = int(y_str.replace(',', ''))
                    ans.append([title, rating, num_votes, year])
                except Exception:
                    # Ignore any errors.
                    continue
            print('ans len: ' + str(len(ans)))
            imdb_df = pd.DataFrame(ans, columns=['title', 'rating', 'num_votes', 'year'])
            if n == 0:
                imdb_df.to_csv('data/imdb.csv', index=False, mode='a', header=True)
            else:
                imdb_df.to_csv('data/imdb.csv', index=False, mode='a', header=False)
            next_link = html_soup.find('a', class_='lister-page-next next-page')
            print(next_link)
            contains_next = (next_link is not None)
            if contains_next:
                url = base_url + next_link['href']
            n += 1

            # IMDB has 98,195 titles displaying 250 per page (approx 390 pages)
            if n == 390:
                break

def analyze_imdb():
    imdb_df = pd.read_csv('data/imdb.csv', header=0, converters={'rating':float, 'num_votes':int, 'year':int})
    print(imdb_df.head())
    print("\n q1.1 find the 20 most popular movies with a rank more than 8.0")
    rating_over_8 = imdb_df.loc[imdb_df['rating'] > 8.0]
    rating_over_8 = rating_over_8.sort_values(by=['num_votes'], ascending=False)
    rating_over_8.reset_index(drop=True, inplace=True)
    print(rating_over_8.head(20))
    print("\n q1.2 find the 20 best rated movies with over 40,000 votes in the 2000s (year >= 2000)")
    votes_over_40k = imdb_df.loc[(imdb_df['num_votes'] > 40000) & (imdb_df['year'] >= 2000)]
    votes_over_40k = votes_over_40k.sort_values(by=['rating'], ascending=False)
    votes_over_40k.reset_index(drop=True, inplace=True)
    print(votes_over_40k.head(20))
    print("\n q2.1 find the average rank of the 10 most popular movies between 2000-2009 (inclusive)")
    movies_2000s = imdb_df.loc[(imdb_df['year'] >= 2000) & (imdb_df['year'] <= 2009)]
    movies_2000s.reset_index(drop=True, inplace=True)
    movies_2000s = movies_2000s.sort_values(by=['num_votes'], ascending=False)
    avg_rating = movies_2000s.iloc[:10]['rating'].mean()
    print('average rating: ' + str(avg_rating))
    print("\n q2.2 find the year in the 1900s when the average rank increased the most, compared to the previous year")
    votes_over_1000 = imdb_df.loc[imdb_df['num_votes'] > 1000]
    movies_1900s = votes_over_1000.loc[(votes_over_1000['year'] >= 1900) & (votes_over_1000['year'] < 2000)]
    movies_1900s.reset_index(drop=True, inplace=True)
    gb = movies_1900s.groupby(['year'])['rating']
    mean_per_year = gb.mean()
    mean_per_year = pd.DataFrame(mean_per_year).reset_index()
    mean_per_year.columns = ['year', 'avg rating']
    mpy_diff = mean_per_year['avg rating'].diff()
    print('Year with greatest increase: ' +  str(int(mean_per_year.iloc[mpy_diff.idxmax()]['year'])))
    print('\n q2.3 find the expected average rank for 2013 using linear regression')
    gb = votes_over_1000.groupby(['year'])['rating']
    mean_per_year = gb.mean()
    mean_per_year = pd.DataFrame(mean_per_year).reset_index()
    mean_per_year.columns = ['year', 'avg rating']
    x_vals = np.array(mean_per_year['year'])
    y_vals = np.array(mean_per_year['avg rating'])
    x_vals = x_vals.reshape(-1, 1)
    regr = linear_model.LinearRegression()
    regr.fit(x_vals, y_vals)
    pred_2013 = regr.predict([[2013]])
    act_2013  = list(mean_per_year.loc[mean_per_year['year'] == 2013]['avg rating'])
    print('predicted value for 2013: ' + str(pred_2013) + ' actual value for 2013: ' + str(act_2013))
    print('How good is the regression?')
    years = np.array(mean_per_year['year'])
    act_vals = np.array(mean_per_year['avg rating'])
    p_vals = []
    for year in years:
        p_vals.append(regr.predict([[year]]))
    r2_value  = metrics.r2_score(act_vals, p_vals)
    mean_sq_err = metrics.mean_squared_error(act_vals, p_vals)
    print('R2 score: ' + str(r2_value))
    print('Mean squared error: ' + str(mean_sq_err))
    print('\n q2.4 find the correlation between rank and votes for each year in the 1900s')
    ratings_and_votes = movies_1900s.drop(columns=['title'], axis=1)
    corr = ratings_and_votes.groupby('year')[['rating', 'num_votes']].corr()
    # print(corr)
    corr = corr['rating'][:, 'num_votes']
    diffs = corr.diff()
    diffs.dropna(inplace=True)
    print(diffs)




def scrape_stock_mini(sym):
    base_url = 'https://finance.yahoo.com/quote/'
    history_url = '/history?'
    period_1_arg = 'period1='
    period_2_arg = 'period2='
    tail_args = 'interval=1d&filter=history&frequency=1d'
    today_ms = math.floor(time.time())
    one_month_ago = datetime.date.today() - datetime.timedelta(days=60)
    oma_ms = int(time.mktime(one_month_ago.timetuple()))
    period_1_arg += str(oma_ms)
    period_2_arg += str(today_ms)
    yahoo_url=base_url + sym + history_url + period_1_arg + '&' + period_2_arg + '&' + tail_args
    response = get(yahoo_url)
    html_soup = BeautifulSoup(response.text, 'html.parser')
    rows = html_soup.find('tbody').find_all('tr')
    closing = []
    date = []
    days = 0
    for row in rows:
        values = row.find_all('td')
        try:
            if(len(values)>2):
                if(values[1].text == '-'):
                   continue
                date.append(values[0].text)
                closing.append(values[4].text)
                days += 1
                if(days == 30):
                    break
        except IndexError:
           continue
    df=pd.DataFrame({'date':date,'closing':closing})
    cur_file_name = 'data/prices/' + sym + '.csv'
    df.to_csv(cur_file_name)


def scrape_stock():
    url_w = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data=[]
    response=get(url_w)
    soup=BeautifulSoup(response.text,'html.parser')
    data=soup.find('tbody').find_all('tr')
    tickers = []
    for x in data[1:]:
        temp = x.find_all('td')
        tickers.append(temp[1].text)
    p = Pool(10)
    p.map(scrape_stock_mini, tickers)
    p.close()
    p.join()

def analyze_stock():
    print('q2.5 identify the stock most correlated with MMM stockprice')
    stocks = pd.DataFrame()
    data_dir = 'data/prices/'
    pathlist = Path(data_dir).glob('*.csv')
    for p in pathlist:
        name = str(p).split('\\')[-1].split('.')[0]
        temp_df = pd.read_csv(p, index_col=0, header=0)
        closing_price = np.array(temp_df['closing'])
        stocks[name] = pd.Series(closing_price)
    stock_name = 'MMM'
    corr = stocks.corr()
    print(corr.head())
    mmm_corr = corr[stock_name]
    mmm_corr.drop(index=[stock_name], inplace=True)
    print('Max. correlation with ' + stock_name + ' -> ' + mmm_corr.idxmax() + ': ' + str(mmm_corr.max()))

def scatter(data_list, title='Title', label='Data', x_label='X', y_label='Y', file_name=None):
    chart = pygal.XY(stroke=False, title=title)
    chart.add(label, data_list)
    chart.x_title = x_label
    chart.y_title = y_label
    if file_name is None:
        file_name = 'scatter-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.svg'
    chart.render_to_file('images/' + file_name)


def svg_plot():
    print('\n q1 Bubble chart in SVG')
    bubble_chart = pygal.Dot(title = 'Sample Bubble Chart')
    bubble_chart.x_labels = ['sample1', 'sample2', 'sample3']
    bubble_chart.y_labels = [0,10,20,30,40,50,60]
    bubble_chart.add('10', [0,0,34])
    bubble_chart.add('30', [0,89,0])
    bubble_chart.add('50', [51,0,0])
    bubble_chart.render_to_file('images/bubble_chart.svg')
    print('\n q2 Draw a scatterplot of rank vs votes for every movie with at least 10,000 votes')
    imdb_df = pd.read_csv('data/imdb.csv', header=0, converters={'rating': float, 'num_votes': int, 'year': int})
    votes_over_10000 = imdb_df.loc[imdb_df['num_votes'] >= 10000]
    votes_over_10000.reset_index()
    mean_votes = votes_over_10000['num_votes'].mean()
    r_and_nv = votes_over_10000[['num_votes', 'rating']]
    r_and_nv = list(map(tuple, r_and_nv.values))
    scatter(r_and_nv[:150], title='IMDB #votes vs. Rating', label='',
                     x_label='Number of votes', y_label='Rating', file_name='scatter_IMDB.svg')
    # r_and_nv = list(map(lambda x: (x[0]/mean_votes, x[1]), r_and_nv))
    # scatter(r_and_nv[:150], title='IMDB #votes vs. Normalized Rating', label='',
    #                  x_label='Number of votes', y_label='Rating', file_name='scatter_post_norm.svg')
    print('\n q3 Draw a correlation matrix of any 30 stocks on the Sensex')
    stocks = pd.DataFrame()
    data_dir = 'data/prices/'
    pathlist = Path(data_dir).glob('*.csv')
    for p in pathlist:
       name = str(p).split('\\')[-1].split('.')[0]
       temp_df = pd.read_csv(p, index_col=0, header=0)
       closing_price = np.array(temp_df['closing'])
       stocks[name] = pd.Series(closing_price)
    print(stocks.head())
    stocks = stocks.sample(30, axis=1)
    corr = stocks.corr()
    colormap = sns.diverging_palette(10, 150, as_cmap=True)
    corr_plot = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=colormap)
    fig = corr_plot.get_figure()
    fig.savefig('images/corr.svg')

def get_max_from_dict(d: dict) -> int:
    ans = max(d)
    if type(ans) == int:
        return ans
    else:
        ans = d[ans]
        if type(ans) == int:
            return ans
        else:
            return 0


def draw_barplot(values: dict, svg_file: str, title: str, x_label='x', y_label='y'):
    loader = template.Loader('.')
    bar_width, x_pos = 18.7, 1
    bar_gap = bar_width / 3
    width, height = (len(values) * bar_width) + ((len(values) - 1) * bar_gap), get_max_from_dict(values) - 75
    translate_y = -height / 10000
    html = loader.load('bar.html').generate(values=values,
                                            bar_width=bar_width,
                                            bar_gap=bar_gap,
                                            x_pos=x_pos,
                                            width=width,
                                            height=height,
                                            translate_y=translate_y,
                                            x_label=x_label,
                                            y_label=y_label,
                                            title=title)
    f = open(svg_file, 'w')
    f.write(html.decode('utf-8'))
    f.close()


def plot_templates():
    print('\n Draw bar graph of the number of movies by year since 1900')
    imdb_df = pd.read_csv('data/imdb.csv', header=0, converters={'rating': float, 'num_votes': int, 'year': int})
    imdb_df.drop(['title', 'rating', 'num_votes'], axis=1, inplace=True)
    gb = imdb_df.groupby(['year'])['year'].count()
    movies = gb.to_dict()
    draw_barplot(values=movies,
                          svg_file='images/bar-template.html',
                          title='Number of movies per year.',
                          x_label='Year',
                          y_label='Number of movies')
    path_pdf = 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'
    config = pdfkit.configuration(wkhtmltopdf=path_pdf)
    pdfkit.from_file('images/bar-template.html', 'images/bar-template.pdf', configuration=config)






if __name__ == '__main__':
##   warnings.filterwarnings("ignore")  
#Learning python
##   print("q1. multiples of 3 less under 20")    
##   multiplesof3()
##   print("\n q2. print palindromic numbers under 1000")
##   for i in range (1000):
##           is_pal(str(i), str(i))
##   print("\n q3. number of characters, words and lines in a file")
##   analyze_file("D:\\MSIT\\DADV\\datavisualization\\project\\data\\diamond.txt")
##   print("\n q4. Analyzing salaries")
##   analyze_data()

#Handling BigData
##    print("\n Getting data from IMDB website")
##    scrape_imdb()
    print("\n analyzing IMDB data")
    analyze_imdb()
##    #print("\n Getting data from yahoo website")
##    #scrape_stock()
##    print('\n analyzing stock data')
##    analyze_stock()

#Vector Graphics
    # print('\n using SVG')
    # svg_plot()

#tornado templates
##    print('\n Plot using tornado templates')
##    plot_templates()




    
    

