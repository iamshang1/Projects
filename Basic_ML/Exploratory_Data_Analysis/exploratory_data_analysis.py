import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Forbes2000 = pd.read_csv("Forbes2000.csv", sep=',', usecols=range(0,9))
Forbes2000 = Forbes2000[1:]

#companies by market value and profits
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
marketv = axes[0].hist(np.array(Forbes2000['Market Value'].astype(float)), 50, range=[0,100], facecolor='green', alpha=0.5)
axes[0].set_title('2014 Forbes 2000 Company Market Values')
axes[0].set_ylabel('# of Companies')
axes[0].set_xlabel('Market Value in Billion $')
axes[0].set_xticks(np.arange(0,101,10))

profits = axes[1].hist(np.array(Forbes2000['Profits'].astype(float)), 50, range=[-5,15], facecolor='green', alpha=0.5)
axes[1].set_title('2014 Forbes 2000 Company Profits')
axes[1].set_ylabel('# of Companies')
axes[1].set_xlabel('Profit in Billion $')
axes[1].set_xticks(np.arange(-4,15,2))

plt.savefig('f1.png')
plt.show()

#separate into sectors
Financials = Forbes2000[Forbes2000.Sector=="Financials"]
Energy = Forbes2000[Forbes2000.Sector=="Energy"]
Industrials = Forbes2000[Forbes2000.Sector=="Industrials"]
IT = Forbes2000[Forbes2000.Sector=="Information Technology"]
ConsumerD = Forbes2000[Forbes2000.Sector=="Consumer Discretionary"]
ConsumerS = Forbes2000[Forbes2000.Sector=="Consumer Staples"]
Health = Forbes2000[Forbes2000.Sector=="Health Care"]
Utilities = Forbes2000[Forbes2000.Sector=="Utilities"]
Telecom = Forbes2000[Forbes2000.Sector=="Telecommunication Services"]
Materials = Forbes2000[Forbes2000.Sector=="Materials"]

#companies by sector
xnames = ['Financials', 'Energy', 'Industrials', 'Information Tech.', 'Cons. Discretionary', 'Cons. Staples', 'Health Care', 'Utilities', 'Telecommunications', 'Materials']
colors = ['lightgreen', 'cornflowerblue', 'lightgrey', 'steelblue', 'plum', 'sandybrown', 'tomato', 'silver', 'violet', 'skyblue']
plt.figure(figsize=(12, 8))
plt.pie([sector.count()[0] for sector in [Financials, Energy, Industrials, IT, ConsumerD, ConsumerS, Health, Utilities, Telecom, Materials]], labels=xnames, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.title("Forbes 2000 Companies by Sector", y=1.08)
plt.savefig('f2.png')
plt.show()

#market value and profits by sector
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))

marketv = axes[0].boxplot([np.array(sector['Market Value'].astype(float)) for sector in [Financials, Energy, Industrials, IT, ConsumerD, ConsumerS, Health, Utilities, Telecom, Materials]], showmeans=True)
axes[0].set_ylabel('Market Value in Billion $')
axes[0].set_ylim(0, 200)
axes[0].set_title('2014 Forbes 2000 Market Value by Sector')
axes[0].set_yticks(np.arange(0,200,10))
axes[0].set_xticklabels(xnames, rotation=45, fontsize=8, ha="right")
axes[0].yaxis.grid(True, linestyle='-', color='lightgrey', alpha=0.5)

profits = axes[1].boxplot([np.array(sector.Profits.astype(float)) for sector in [Financials, Energy, Industrials, IT, ConsumerD, ConsumerS, Health, Utilities, Telecom, Materials]], showmeans=True)
axes[1].set_ylabel('Profits in Billion $')
axes[1].set_ylim(-4, 20)
axes[1].set_title('2014 Forbes 2000 Profits by Sector')
axes[1].set_yticks(np.arange(-4,20,2))
axes[1].set_xticklabels(xnames, rotation=45, fontsize=8, ha="right")
axes[1].yaxis.grid(True, linestyle='-', color='lightgrey', alpha=0.5)

plt.savefig('f3.png')
plt.show()

#separate by continent
NA = Forbes2000[Forbes2000.Continent=="North America"]
SA = Forbes2000[Forbes2000.Continent=="South America"]
Europe = Forbes2000[Forbes2000.Continent=="Europe"]
Asia = Forbes2000[Forbes2000.Continent=="Asia"]
Australia = Forbes2000[Forbes2000.Continent=="Australia"]
Africa = Forbes2000[Forbes2000.Continent=="Africa"]

#companies by continent
xnames = ['North America', 'South America', 'Europe', 'Australia', 'Asia', 'Africa']
colors = ['cornflowerblue', 'tomato', 'violet', 'gold', 'palegreen', 'sandybrown']
plt.figure(figsize=(12, 8))
plt.pie([continent.count()[0] for continent in [NA, SA, Europe, Australia, Asia, Africa]], labels=xnames, colors=colors, autopct='%1.1f%%', shadow=True, startangle=30)
plt.axis('equal')
plt.title("Forbes 2000 Companies by Continent", y=1.08)
plt.savefig('f4.png')
plt.show()

#market value and profits by continent
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))

marketv = axes[0].boxplot([np.array(continent['Market Value'].astype(float)) for continent in [NA, SA, Europe, Australia, Asia, Africa]], showmeans=True)
axes[0].set_ylabel('Market Value in Billion $')
axes[0].set_ylim(0, 300)
axes[0].set_title('2014 Forbes 2000 Market Value by Continent')
axes[0].set_yticks(np.arange(0,300,20))
axes[0].set_xticklabels(xnames, rotation=45, fontsize=8, ha="right")
axes[0].yaxis.grid(True, linestyle='-', color='lightgrey', alpha=0.5)

profits = axes[1].boxplot([np.array(continent.Profits.astype(float)) for continent in [NA, SA, Europe, Australia, Asia, Africa]], showmeans=True)
axes[1].set_ylabel('Profits in Billion $')
axes[1].set_ylim(-5, 30)
axes[1].set_title('2014 Forbes 2000 Profits by Continent')
axes[1].set_yticks(np.arange(-5,30,5))
axes[1].set_xticklabels(xnames, rotation=45, fontsize=8, ha="right")
axes[1].yaxis.grid(True, linestyle='-', color='lightgrey', alpha=0.5)

plt.savefig('f5.png')
plt.show()

#relationship vetween profits and market value
plt.figure(figsize=(12, 8))
marketv = np.array(Forbes2000['Market Value'].astype(float))
profits = np.array(Forbes2000['Profits'].astype(float))
plt.scatter(marketv, profits, alpha=0.5)
plt.title("Relationship Between Market Value and Profits", y=1.08)
plt.xlabel('Market Value in Billion $')
plt.ylabel('Profit in Billion $')
plt.xlim(-20, 500)
plt.savefig('f6.png')
plt.show()