
# coding: utf-8

# ## 1. Introduction to Baby Names Data
# <blockquote>
#   <p>What’s in a name? That which we call a rose, By any other name would smell as sweet.</p>
# </blockquote>
# <p>In this project, we will explore a rich dataset of first names of babies born in the US, that spans a period of more than 100 years! This suprisingly simple dataset can help us uncover so many interesting stories, and that is exactly what we are going to be doing. </p>
# <p>Let us start by reading the data.</p>

# In[56]:


# Import modules
import numpy as np
import pandas as pd

# Read names into a dataframe: bnames
bnames = pd.read_csv('datasets/names.csv.gz')
bnames.head()
bnames.info()


# In[57]:


get_ipython().run_cell_magic('nose', '', 'def test_bnames_exists():\n    """bnames is defined."""\n    assert \'bnames\' in globals(), "You should have defined a variable named bnames"\n# bnames is a dataframe with 1891894 rows and 4 columns\ndef test_bnames_dataframe():\n    """bnames is a DataFrame with 1891894 rows and 4 columns"""\n    import pandas as pd\n    assert isinstance(bnames, pd.DataFrame)\n    assert bnames.shape[0] == 1891894, "Your  DataFrame, bnames, should contain 1891984 rows"\n    assert bnames.shape[1] == 4, "Your DataFrame, bnames, should contain 4 columns"\n\n# bnames has column names [\'name\', \'sex\', \'births\', \'year\']\ndef test_bnames_colnames():\n    """bnames has column names [\'name\', \'sex\', \'births\', \'year\']"""\n    colnames = [\'name\', \'sex\', \'births\', \'year\']\n    assert all(name in bnames for name in colnames), "Your DataFrame, bnames, should have columns named name, sex, births and year"')


# ## 2. Exploring Trends in Names
# <p>One of the first things we want to do is to understand naming trends. Let us start by figuring out the top five most popular male and female names for this decade (born 2011 and later). Do you want to make any guesses? Go on, be a sport!!</p>

# In[58]:


# bnames_top5: A dataframe with top 5 popular male and female names for the decade
bnames_2010 = bnames.loc[bnames['year'] > 2010]
bnames_2010.head()

# Aggregate names with respective to sex, show births 
bnames_2010_agg = bnames_2010.groupby(['name', 'sex'], as_index=False)['births'].sum()

# Sort values for by sex and births. Groupby sex, make a new dataframe and show twice (for the two sex classes) the top 5 names, sex and births
# Without the .head().reset_index() the groupby would be a groupby object rather than a new dataframe
bnames_top5 = bnames_2010_agg.sort_values(['sex', 'births'], ascending=[True, False]).groupby('sex').head().reset_index(drop=True)
bnames_top5


# In[59]:


get_ipython().run_cell_magic('nose', '', 'def test_bnames_top5_exists():\n    """bnames_top5 is defined."""\n    assert \'bnames_top5\' in globals(), \\\n      "You should have defined a variable named bnames_top5."\n\ndef test_bnames_top5_df():\n    """Output is a DataFrame with 10 rows and 3 columns."""\n    assert bnames_top5.shape == (10, 3), \\\n      "Your DataFrame, bnames_top5, should have 10 rows and 3 columns."\n\ndef test_bnames_top5_df_colnames():\n    """Output has column names: name, sex, births."""\n    assert all(name in bnames_top5 for name in [\'name\', \'sex\', \'births\']), \\\n      "Your DataFrame, bnames_top5 should have columns named name, sex, births."\n\ndef test_bnames_top5_df_contains_names():\n    """Output has the follwing female names: Emma, Sophia, Olivia, Isabella, Ava"""\n    target_names = [\'Emma\', \'Sophia\', \'Olivia\', \'Isabella\', \'Ava\']\n    assert set(target_names).issubset(bnames_top5[\'name\']), \\\n      "Your DataFrame, bnames_top5 should contain the female names: Emma, Sophia, Olivia, Isabella, Ava"\n\ndef test_bnames_top5_df_contains_female_names():\n    """Output has the following male names: Noah, Mason, Jacob, Liam, William"""\n    target_names = [\'Noah\', \'Mason\', \'Jacob\', \'Liam\', \'William\']\n    assert set(target_names).issubset(bnames_top5[\'name\']), \\\n      "Your DataFrame, bnames_top5 should contain the male names: Noah, Mason, Jacob, Liam, William"')


# ## 3. Proportion of Births
# <p>While the number of births is a useful metric, making comparisons across years becomes difficult, as one would have to control for population effects. One way around this is to normalize the number of births by the total number of births in that year.</p>

# In[60]:


bnames2 = bnames.copy()

# Compute the proportion of births by year and add it as a new column
# Transform function lets apply the groupby without changing the shape to enable instant broadcasting
births_per_year = bnames2.groupby('year', as_index=True)['births'].transform(lambda x: x.sum())
bnames2['prop_births'] = bnames2['births']/births_per_year

bnames2.head()


# In[61]:


get_ipython().run_cell_magic('nose', '', 'def test_bnames2_exists():\n    """bnames2 is defined."""\n    assert \'bnames2\' in globals(),\\\n      "You should have defined a variable named bnames2."\n    \ndef test_bnames2_dataframe():\n    """bnames2 is a DataFrame with 1891894 rows and 5 columns"""\n    import pandas as pd\n    assert isinstance(bnames2, pd.DataFrame)\n    assert bnames2.shape[1] == 5,\\\n      "Your DataFrame, bnames2, should have 5 columns"\n    assert bnames2.shape[0] == 1891894,\\\n      "Your DataFrame, bnames2,  should have 1891894 rows"\n\n\ndef test_bnames2_colnames():\n    """bnames2 has column names [\'name\', \'sex\', \'births\', \'year\', \'prop_births\']"""\n    colnames = [\'name\', \'sex\', \'births\', \'year\', \'prop_births\']\n    assert all(name in bnames2 for name in colnames),\\\n      "Your DataFrame, bnames2, should have column names \'name\', \'sex\', \'births\', \'year\', \'prop_births\'"')


# ## 4. Popularity of Names
# <p>Now that we have the proportion of births, let us plot the popularity of a name through the years. How about plotting the popularity of the female names <code>Elizabeth</code>, and <code>Deneen</code>, and inspecting the underlying trends for any interesting patterns!</p>

# In[62]:


# Set up matplotlib for plotting in the notebook.
%matplotlib inline
import matplotlib.pyplot as plt

def plot_trends(name, sex):
        births = bnames[(bnames['name'] == name) & (bnames['sex'] == sex)][['year', 'births']]
        births.plot(x = 'year', y = 'births')
        plt.xlabel('Year')
        plt.ylabel('Number of births')
        plt.title('Popularity of name ' + name + ' from 1880 to 2016')
        return plt.show()


# Plot trends for Elizabeth and Deneen 
plot_trends('Elizabeth', 'F')
plot_trends('Deneen', 'F')
    
# How many times did these female names peak?
num_peaks_elizabeth = 3
num_peaks_deneen    = 1


# In[63]:


get_ipython().run_cell_magic('nose', '', 'def test_peaks_elizabeth():\n    """The name Elizabeth peaks 3 times."""\n    assert num_peaks_elizabeth == 3, \\\n      "The name Elizabeth peaks 3 times"\n    \ndef test_peaks_deneen():\n    """The name Deneen peaks 1 time."""\n    assert num_peaks_deneen == 1, \\\n      "The name Deneen peaks only once"')


# ## 5. Trendy vs. Stable Names
# <p>Based on the plots we created earlier, we can see that <strong>Elizabeth</strong> is a fairly stable name, while <strong>Deneen</strong> is not. An interesting question to ask would be what are the top 5 stable and top 5 trendiest names. A stable name is one whose proportion across years does not vary drastically, while a trendy name is one whose popularity peaks for a short period and then dies down. </p>
# <p>There are many ways to measure trendiness. A simple measure would be to look at the maximum proportion of births for a name, normalized by the sume of proportion of births across years. For example, if the name <code>Joe</code> had the proportions <code>0.1, 0.2, 0.1, 0.1</code>, then the trendiness measure would be <code>0.2/(0.1 + 0.2 + 0.1 + 0.1)</code> which equals <code>0.5</code>.</p>
# <p>Let us use this idea to figure out the top 10 trendy names in this data set, with at least a 1000 births.</p>

# In[64]:


# top10_trendy_names | A Data Frame of the top 10 most trendy names
names = pd.DataFrame()
n_s_grouped = bnames.groupby(['name', 'sex'])
names['total'] = n_s_grouped['births'].sum()
names.head()

names['max'] = bnames.groupby(['name', 'sex'])['births'].max()
names.head()

names['trendiness'] = names['max'].div(names['total'])
names.head()

top10_trendy_names = names.loc[names['total'] > 1000].sort_values('trendiness', ascending=False).head(10).reset_index()
top10_trendy_names


# In[65]:


get_ipython().run_cell_magic('nose', '', 'def test_top10_trendy_names_exists():\n    """top10_trendy_names is defined"""\n    assert \'top10_trendy_names\' in globals(), \\\n      "You should have defined a variable namedtop10_trendy_names."\ndef test_top10_trendy_df():\n    """top10_trendy_names is a dataframe with 10 rows and 5 columns."""\n    assert top10_trendy_names.shape == (10, 5), \\\n      "Your data frame, top10_trendy_names, should have 10 rows and 5 columns."\n\ndef test_top10_trendy_df_colnames():\n    """top10_trendy_names has column names: name, sex, births, max and trendiness"""\n    assert all(name in top10_trendy_names for name in [\'name\', \'sex\', \'total\', \'max\', \'trendiness\']), \\\n       "Your data frame, top10_trendy_names, should have column names: name, sex, births, max and trendiness"\n\ndef test_top10_trendy_df_contains_female_names():\n    """top10_trendy_names has the follwing female names: Royalty, Kizzy, Aitana, Deneen, Moesha, Marely, Tennille, Kadijah"""\n    target_names = [\'Royalty\', \'Kizzy\', \'Aitana\', \'Deneen\', \'Moesha\', \'Marely\', \'Tennille\', \'Kadijah\']\n    assert set(target_names).issubset(top10_trendy_names[\'name\']), \\\n      "Your data frame, top10_trendy_names, should have female names: Royalty, Kizzy, Aitana, Deneen, Moesha, Marely, Tennille, Kadijah."\n\ndef test_top10_trendy_df_contains_male_names():\n    """top10_trendy_names has the following male names: Christop, Kanye"""\n    target_names = [\'Christop\', \'Kanye\']\n    assert set(target_names).issubset(top10_trendy_names[\'name\']), \\\n      "Your data frame, top10_trendy_names, should have male names: Christop, Kanye"')


# ## 6. Bring in Mortality Data
# <p>So, what more is in a name? Well, with some further work, it is possible to predict the age of a person based on the name (Whoa! Really????). For this, we will need actuarial data that can tell us the chances that someone is still alive, based on when they were born. Fortunately, the <a href="https://www.ssa.gov/">SSA</a> provides detailed <a href="https://www.ssa.gov/oact/STATS/table4c6.html">actuarial life tables</a> by birth cohorts.</p>
# <table>
# <thead>
# <tr>
# <th style="text-align:right;">year</th>
# <th style="text-align:right;">age</th>
# <th style="text-align:right;">qx</th>
# <th style="text-align:right;">lx</th>
# <th style="text-align:right;">dx</th>
# <th style="text-align:right;">Lx</th>
# <th style="text-align:right;">Tx</th>
# <th style="text-align:right;">ex</th>
# <th style="text-align:left;">sex</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td style="text-align:right;">1910</td>
# <td style="text-align:right;">39</td>
# <td style="text-align:right;">0.00283</td>
# <td style="text-align:right;">78275</td>
# <td style="text-align:right;">222</td>
# <td style="text-align:right;">78164</td>
# <td style="text-align:right;">3129636</td>
# <td style="text-align:right;">39.98</td>
# <td style="text-align:left;">F</td>
# </tr>
# <tr>
# <td style="text-align:right;">1910</td>
# <td style="text-align:right;">40</td>
# <td style="text-align:right;">0.00297</td>
# <td style="text-align:right;">78053</td>
# <td style="text-align:right;">232</td>
# <td style="text-align:right;">77937</td>
# <td style="text-align:right;">3051472</td>
# <td style="text-align:right;">39.09</td>
# <td style="text-align:left;">F</td>
# </tr>
# <tr>
# <td style="text-align:right;">1910</td>
# <td style="text-align:right;">41</td>
# <td style="text-align:right;">0.00318</td>
# <td style="text-align:right;">77821</td>
# <td style="text-align:right;">248</td>
# <td style="text-align:right;">77697</td>
# <td style="text-align:right;">2973535</td>
# <td style="text-align:right;">38.21</td>
# <td style="text-align:left;">F</td>
# </tr>
# <tr>
# <td style="text-align:right;">1910</td>
# <td style="text-align:right;">42</td>
# <td style="text-align:right;">0.00332</td>
# <td style="text-align:right;">77573</td>
# <td style="text-align:right;">257</td>
# <td style="text-align:right;">77444</td>
# <td style="text-align:right;">2895838</td>
# <td style="text-align:right;">37.33</td>
# <td style="text-align:left;">F</td>
# </tr>
# <tr>
# <td style="text-align:right;">1910</td>
# <td style="text-align:right;">43</td>
# <td style="text-align:right;">0.00346</td>
# <td style="text-align:right;">77316</td>
# <td style="text-align:right;">268</td>
# <td style="text-align:right;">77182</td>
# <td style="text-align:right;">2818394</td>
# <td style="text-align:right;">36.45</td>
# <td style="text-align:left;">F</td>
# </tr>
# <tr>
# <td style="text-align:right;">1910</td>
# <td style="text-align:right;">44</td>
# <td style="text-align:right;">0.00351</td>
# <td style="text-align:right;">77048</td>
# <td style="text-align:right;">270</td>
# <td style="text-align:right;">76913</td>
# <td style="text-align:right;">2741212</td>
# <td style="text-align:right;">35.58</td>
# <td style="text-align:left;">F</td>
# </tr>
# </tbody>
# </table>
# <p>You can read the <a href="https://www.ssa.gov/oact/NOTES/as120/LifeTables_Body.html">documentation for the lifetables</a> to understand what the different columns mean. The key column of interest to us is <code>lx</code>, which provides the number of people born in a <code>year</code> who live upto a given <code>age</code>. The probability of being alive can be derived as <code>lx</code> by 100,000. </p>
# <p>Given that 2016 is the latest year in the baby names dataset, we are interested only in a subset of this data, that will help us answer the question, "What percentage of people born in Year X are still alive in 2016?" </p>
# <p>Let us use this data and plot it to get a sense of the mortality distribution!</p>

# In[66]:


# Read lifetables from datasets/lifetables.csv
lifetables = pd.read_csv('datasets/lifetables.csv')

# Extract subset relevant to those alive in 2016
lifetables_2016 = lifetables[lifetables['year'] + lifetables['age'] == 2016]

# Plot the mortality distribution: year vs. lx
plt.title('Mortality Rate Curve 1900 - 2020')
plt.xlabel('Birth Year')
plt.ylabel('Probability of being alive as %')
plt.yticks(range(0,100,10))
plt.plot(lifetables_2016['year'], lifetables_2016['lx']/1000)


# In[67]:


get_ipython().run_cell_magic('nose', '', 'def test_lifetables_2016_exists():\n    """lifetables_2016 is defined"""\n    assert \'lifetables_2016\' in globals(), \\\n      "You should have defined a variable named lifetables_2016."\ndef test_lifetables_2016_df():\n    """Output is a DataFrame with 24 rows and 9 columns."""\n    assert lifetables_2016.shape == (24, 9), \\\n      "Your DataFrame, lifetables_2016, should have 24 rows and 9 columns."\n\ndef test_lifetables_2016_df_colnames():\n    """Output has column names: year, age, qx, lx, dx, Lx, Tx, ex, sex"""\n    assert all(name in lifetables_2016 for name in [\'year\', \'age\', \'qx\', \'lx\', \'dx\', \'Lx\', \'Tx\', \'ex\', \'sex\']), \\\n      "Your DataFrame, lifetables_2016, should have columns named: year, age, qx, lx, dx, Lx, Tx, ex, sex."\n\ndef test_lifetables_2016_df_year_plus_age():\n    """Output has the year + age = 2016"""\n    assert all(lifetables_2016.year + lifetables_2016.age - 2016 == 0), \\\n      "The `year` column and `age` column in `lifetables_2016` should sum up to 2016."')


# ## 7. Smoothen the Curve!
# <p>We are almost there. There is just one small glitch. The cohort life tables are provided only for every decade. In order to figure out the distribution of people alive, we need the probabilities for every year. One way to fill up the gaps in the data is to use some kind of interpolation. Let us keep things simple and use linear interpolation to fill out the gaps in values of <code>lx</code>, between the years <code>1900</code> and <code>2016</code>.</p>

# In[68]:


# Create smoothened lifetable_2016_s by interpolating values of lx
year = np.arange(1900, 2016)

mf = {"M": pd.DataFrame(), "F": pd.DataFrame()}

for sex in ["M", "F"]: 
    d = lifetables_2016[lifetables_2016['sex'] == sex][["year", "lx"]]
    mf[sex] = d.set_index('year').reindex(year).interpolate().reset_index()
    mf[sex]['sex'] = sex 
    
lifetable_2016_s = pd.concat(mf, ignore_index = True)
lifetable_2016_s


# In[69]:


get_ipython().run_cell_magic('nose', '', 'def test_lifetable_2016_s_exists():\n    """lifetable_2016_s is defined"""\n    assert \'lifetable_2016_s\' in globals(), \\\n      "You should have defined a variable named lifetable_2016_s."\ndef test_lifetables_2016_s_df():\n    """lifetable_2016_s is a dataframe with 232 rows and 3 columns."""\n    assert lifetable_2016_s.shape == (232, 3), \\\n      "Your DataFrame, lifetable_2016_s, should have 232 rows and 3 columns."\n\ndef test_lifetable_2016_s_df_colnames():\n    """lifetable_2016_s has column names: year, lx, sex"""\n    assert all(name in lifetable_2016_s for name in [\'year\', \'lx\', \'sex\']), \\\n      "Your DataFrame, lifetable_2016_s, should have columns named: year, lx, sex."')


# ## 8. Distribution of People Alive by Name
# <p>Now that we have all the required data, we need a few helper functions to help us with our analysis. </p>
# <p>The first function we will write is <code>get_data</code>,which takes <code>name</code> and <code>sex</code> as inputs and returns a data frame with the distribution of number of births and number of people alive by year.</p>
# <p>The second function is <code>plot_name</code> which accepts the same arguments as <code>get_data</code>, but returns a line plot of the distribution of number of births, overlaid by an area plot of the number alive by year.</p>
# <p>Using these functions, we will plot the distribution of births for boys named <strong>Joseph</strong> and girls named <strong>Brittany</strong>.</p>

# In[70]:


def get_data(name, sex):
    born = bnames[(bnames['name'] == name) & (bnames['sex'] == sex)][['year', 'births']]
    alive = lifetable_2016_s[lifetable_2016_s['sex'] == sex]
    combined = pd.concat([born.set_index('year'), alive.set_index('year')], axis=1)
    x = combined[(combined['lx'].notnull()) & (combined['births'].notnull())].reset_index()
    x['name'] = name
    x['n_alive'] = x['lx']*x['births']/100000
    rearranged = x[['name', 'sex', 'births', 'year', 'lx', 'n_alive']]
    return rearranged


def plot_data(name, sex):
    data = get_data(name, sex)
    fig, ax1 = plt.subplots()
    plt.title('Births and Number of ' + str(name) + ' Alive in US 2016')
    ax1.set_ylabel('Births')
    ax1.set_xlabel('Year')
    ax1.plot(data['year'], data['births'], label='births', c='blue')
    ax2 = ax1.twinx()
    ax2.plot(data['year'], data['n_alive'], label='% - Alive', c='black')
    ax2.set_ylabel('Percentage of births still alive in 2016')
    ax2.axes.yaxis.set_ticklabels(range(0, 100, 10))
    plt.legend(loc='best')
    return plt.show()
    
# Plot the distribution of births and number alive for Joseph and Brittany
Joseph = plot_data('Joseph', 'M')
Brittany = plot_data('Brittany', 'F')


# In[71]:


get_ipython().run_cell_magic('nose', '', 'joseph = get_data(\'Joseph\', \'M\')\ndef test_joseph_df():\n    """get_data(\'Joseph\', \'M\') is a dataframe with 116 rows and 6 columns."""\n    assert joseph.shape == (116, 6), \\\n      "Running  get_data(\'Joseph\', \'M\') should return a data frame with 116 rows and 6 columns."\n\ndef test_joseph_df_colnames():\n    """get_data(\'Joseph\', \'M\') has column names: name, sex, births, year, lx, n_alive"""\n    assert all(name in lifetable_2016_s for name in [\'year\', \'lx\', \'sex\']), \\\n      "Running  get_data(\'Joseph\', \'M\') should return a data frame with column names: name, sex, births, year, lx, n_alive"')


# ## 9. Estimate Age
# <p>In this section, we want to figure out the probability that a person with a certain name is alive, as well as the quantiles of their age distribution. In particular, we will estimate the age of a female named <strong>Gertrude</strong>. Any guesses on how old a person with this name is? How about a male named <strong>William</strong>?</p>

# In[72]:


# Import modules
from wquantiles import quantile

# Function to estimate age quantiles
def estimate_age(name, sex):
    data = get_data(name, sex)
    qs = [.25, .5, .75]
    quantiles = [2016 - int(quantile(data.year, data.n_alive, q)) for q in qs]
    result = dict(zip(['q25', 'q50', 'q75'], quantiles))
    result['p_alive'] = data.n_alive.sum()/data.births.sum()*100
    result['sex'] = sex
    result['name'] = name
    return pd.Series(result)

# Estimate the age of Gertrude
Gertrude = estimate_age('Gertrude', 'F')
Gertrude


# In[73]:


get_ipython().run_cell_magic('nose', '', 'gertrude = estimate_age(\'Gertrude\', \'F\')\ndef test_gertrude_names():\n    """Series has indices name, p_alive, q25, q50 and q75"""\n    expected_names = [\'name\', \'p_alive\', \'q25\', \'q50\', \'q75\']\n    assert all(name in gertrude.index.values for name in expected_names), \\\n      "Your function `estimate_age` should return a series with names: name, p_alive, q25, q50 and q75"\n\ndef test_gertrude_q50():\n    """50th Percentile of age for Gertrude is between 75 and 85"""\n    assert ((75 < gertrude[\'q50\']) and (gertrude[\'q50\'] < 85)), \\\n      "The estimated median age for the name Gertrude should be between 75 and 85."')


# ## 10. Median Age of Top 10 Female Names
# <p>In the previous section, we estimated the age of a female named Gertrude. Let's go one step further this time, and compute the 25th, 50th and 75th percentiles of age, and the probability of being alive for the top 10 most common female names of all time. This should give us some interesting insights on how these names stack up in terms of median ages!</p>

# In[74]:


# Create median_ages: DataFrame with Top 10 Female names, 
#    age percentiles and probability of being alive
# bnames_top10: A dataframe with top 10 popular female names of all time

# Aggregate names with respective to sex, show births 
bnames_f_top10 = bnames.groupby(['name', 'sex'], as_index=False)['births'].sum().sort_values(['sex', 'births'], ascending=[True, False]).iloc[:10]
bnames_f_top10

# Run the estimate_age function for all names in the dataframe and concatenate the results to Pandas DataFrame
estimates = pd.concat([estimate_age(name, 'F') for name in bnames_f_top10['name']], axis=1)

# The values are concatenated columnwise, here the rows are transposed to columns and sorted by median age.
median_ages = estimates.transpose().sort_values('q50', ascending=False).reset_index(drop=True)
median_ages



# In[75]:


get_ipython().run_cell_magic('nose', '', 'def test_median_ages_exists():\n    """median_ages is defined"""\n    assert \'median_ages\' in globals(), \\\n      "You should have a variable named median_ages defined."\ndef test_median_ages_df():\n    """median_ages is a dataframe with 10 rows and 6 columns."""\n    assert median_ages.shape == (10, 6), \\\n      "Your DataFrame, median_ages, should have 10 rows and 6 columns"\n\ndef test_median_ages_df_colnames():\n    """median_ages has column names: name, p_alive, q25, q50, q75 and sex"""\n    assert all(name in median_ages for name in [\'name\', \'p_alive\', \'q25\', \'q50\', \'q75\', \'sex\']), \\\n      "Your DataFrame, median_ages, should have columns named: name, p_alive, q25, q50, q75 and sex"')

