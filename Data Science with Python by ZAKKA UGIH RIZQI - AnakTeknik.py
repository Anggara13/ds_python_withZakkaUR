#!/usr/bin/env python
# coding: utf-8

# # CAUTION!
# **Material is compiled by Zakka Ugih Rizqi**
# 
# The data and material contained in this file is reserved.
# 
# *Please do not use without the author’s permission. All rights reserved* - Partial Courtesy to IBM Corporation
# 
# <img src = "https://pbs.twimg.com/media/E5q6ByVUcAYpnmI?format=jpg&name=4096x4096" align = "center" width="460" />
#                                                
#                                                Please say Hello if we meet!
# 

# # 1. Basic Python
# Python is an interpreted language which means we can run the python code from the command line or from its interpreted shell directly without any compilers. Python is also General purpose, Open Source, and Relatively Easy to Use.
# Python language includes many ideas from object-oriented and procedural programming.
# 
# In order to operate python easily, we need Integrated Development Environment (IDE). Jupyter Notebook is one of them. It is an interactive and enables us to run code in smaller chunks called “cells”. In each cell, we can just write the code directly. To run the code, we can click **shift+enter** or **ctrl+control** -> Try it and see the difference. If we want to add blank space in the below, just click **B**. If we forget the complete syntax of function or variable, we can click **tab**. If we forget the parameters or definition of the function we can click **shift+tab** inside the paranthesis such as input(*click shift+tab here*)
# 
# Variable Concept: Variable means we store a value inside something (called variable) that characterizes with "=". Variable name should be created from letter or word, not a number. If variable name should be in sentence, we can use underscore "_", we cannot put space inside a variable name. And remember, Python is case-sentitive that means A is not the same with a.

# ### Simple Input/Output

# In[1]:


# Input function
input("Enter a number:")


# In[2]:


# Put input function to variable A
A=input("Enter your name:")
A


# In[3]:


# Output using simple print function
print(1+1)
print("Hello World")


# In[4]:


# There is also Print Formatting. For example:
print("The winner is {} {} {}".format("Zakka", "Ziki", "Ziba"))
print("The winner is {1} {0} {2}".format("Zakka", "Ziki", "Ziba"))
# or we can use another way
a="Hai"
print(f"The winner is {a} {0} {2}")


# ### Python Operations
# 
# There are 7 operations in Python. But only some of them are usually used:
# 
# 1. Arithmetic Operators
# <img src = "https://www.devopsschool.com/blog/wp-content/uploads/2020/08/arithmetic-operation-in-python.png" align = "center" />
# 2. Comparison (Relational) Operators
# <img src = "https://www.devopsschool.com/blog/wp-content/uploads/2020/08/relational-operator-in-python.png" align = "center" />
# 3. Assignment Operators
# <img src = "https://www.devopsschool.com/blog/wp-content/uploads/2020/08/assignment-operator-in-python.png" align = "center" />
# 4. Logical Operators
# <img src = "https://www.devopsschool.com/blog/wp-content/uploads/2020/08/logical-operator-in-python-1.png" align = "center" />
# 5. Bitwise Operators 
# 6. Membership Operators 
# 7. Identity Operators

# ### Data Structures
# 
# There are 2 types of data structures in python:
# 
# 1. Primitive: Float -> 1.1, Integer -> 1, String -> "Zakka", Boolean -> True (1) or False (0)
# 2. Non-Primitive (Built-in): List -> [1,2,3], Tuple -> (1,2,3), Set -> {1,2,3}, Dictionary -> ["transaction1":100,"transaction2":2000]   
# 
# **Two important concepts in data structures:**
# 
# *Iterable VS Non-Iterable*
# 
# 1. Iterable: String, List, Tuple, Set, Dictionary
# 2. Non-Iterable: Float, Integer, Boolean
# 
# *Mutable VS Imutable*
# 1. Mutable: List, Set, Dictionary
# 2. Immutable: String, Tuple, Float, Integer, Boolean

# In[6]:


# Proof of iterable VS non-iterable
example_of_string='Zakka'
example_of_float=1.5

#Iterable 
for i in example_of_string:
    print(i)
#Non-Iterable 
# for i in example_of_float:
#     print(i)
    
# There should be an error with a float


# In[7]:


# Proof of Mutable VS Immutable
example_of_list=[1,2,3,4,5]
example_of_string='Zakka'

# Mutable
example_of_list[0]=10
print(example_of_list)
# Immutable  
# example_of_string[0]='C'
# print(example_of_string)
# There should be an error with a string


# **String**
# 
# String is text variable that is encapsulated by quotation mark either single quote (') or double quotes (")

# In[8]:


str="I am Good"
print(str)
print(str[0])


# In[9]:


dream_of_me='I will be good data scientist'
print(dream_of_me)


# **LIST**
# 
# A list is created using square brackets with commas separating items [ item, item , item]

# In[10]:


# Simple list
fruits=["Apple","Orange","Papaya"]
print(fruits)
print(fruits[1])


# In[11]:


#List may be nested [ item, [ item, item , item] item , item]
words=["Apple",["Eggs", "Scrambled"],"Papaya"]
print(words[1][0])


# In[12]:


#The range function creates a sequential list of numbers
numbers=list(range(10))
print(numbers)


# In[13]:


#append
nums=[1,2,3]
nums.append(4)
print(nums)


# In[14]:


#len
print(len(nums))


# In[15]:


#insert
nums.insert(0,0)
print(nums)


# In[16]:


#list comprenhension
cube=[i**3 for i in range(5)]
print(cube)


# In[17]:


#Range can be called with two arguments
numbers=list(range(3,8))
print(numbers)


# In[18]:


#Range can be called with three arguments, the third argument determines the interval
numbers=list(range(3,8,2))
print(numbers)


# **Dictionaries**
# 
# Dictionaries are like Lists but with indexed by Keys instead of integers: { Key: item, Key: item, Key: item }
# 
# Curly brackets are used to create Dictionaries. Square brackets are used (as in List) to retrieve items.

# In[19]:


age={"David":34, "Lee": 24, "Ruby":32}
print(age["David"])


# In[20]:


#Keys can be assigned values
age["David"]=22
print(age["David"])


# In[21]:


#To determine if a key is in the dictionary	
print ("David" in age)


# In[22]:


#using get
pairs= {1: "apple", "orange": [2,3,4],True: False,None: "True",}
print(pairs.get("orange"))
print(pairs.get(7))


# In[23]:


#check dictionary using in or not in
nums={1:"one",2:"two",3:"three",}
print (1 in nums)
print(4 not in nums)


# **Tuples**
# 
# Tuples are like Lists but they are immutable. Parentheses are used instead of square brackets for Lists.
# 
# ( item, item , item )
# 
# Tuples can also be created without parenthesis:
# 
# words="dog", "cat", "bird"

# In[24]:


words=("dog", "cat", "bird")
print(words[0])
#tuple does not support item assignment
#words[0]="fish"


# ### If-Else Statement
# Remember: After antecedent/conditional (statement after if), we need to use colon and the indentation after enter.

# In[25]:


if 1>4:
    print(1)
else:
    print("2")


# In[26]:


spam=7
if spam<5:
    print("five")
if spam>8:
    print("eight")


# In[27]:


spam=7
if spam<5:
    print("five")
elif spam>8:
    print("eight") 
else:
    print("seven")    


# ### Looping
# There are 2 types of looping:
# 1. For loop: looping until certain number is reached
# 2. While Loop: looping as long as the statement is True
# 3. Nested Loop: There will be a loop (inner loop) inside a loop (outer loop)

# In[28]:


# For loop will be stopped with certain number is reached
# Use for loop to enumerate list variable
lista=[1,2,3,4,5]
for i in lista:
    print(i)


# In[29]:


# Use for loop to enumerate list variable INSIDE range function
for i in range(5):
    print(i)
# Remember, python always start first number using zero, not one


# In[30]:


words=["hello","world","good"]
for word in words:
   print(word)


# In[31]:


# while loop will be stopped with condition
i=1
while i<5:
   print(i)
   i=i+1
print("finish")


# In[32]:


# nested loop
# outer loop
for i in range(1, 11):
    # inner loop
    for j in range(1, 11):
        # print multiplication
        print(i * j, end=' ')
    print()


# ### Functions
# Functions are defined using the def keyword. Functions must be defined before it is called. The return statement is optional. 
# ```python
# def my_function():
# print(“spam”)
# 
# my_function()
# ```

# In[33]:


# def functions

# function definition with parameters
def calculator_plus(x,y):
    print(x+y)
# function recall with arguments
calculator_plus(2,1)


# In[34]:


# def with return statement
def find_max(x,y):
    if x>=y:
        return x
    else:
        return y
find_max(10,5)


# <hr>
# <h2>FINISH!</h2>
# <p>Congratulations, you have completed your lesson and hands-on of Basic Python. However, there are many things you need to do. You do not need to memorize everything in coding! So, please read here for further explanation <a href="https://docs.python.org/3/" target="_blank">this article</a> to learn how to share your work.
# <hr>

# # 2. Introduction to Numpy
# Keep in mind that numpy (NUMerical PYthon) is here to help with computing
# 
# Compared to the list, Numpy has advantage not only in computation but also flexibility

# In[35]:


import numpy as np
# For example, we want to add each element for each index
a=[1,2,3]
b=[3,4,5]
c=a+b
c


# In[36]:


# In Numpy
np_a=np.array([1,2,3])
np_b=np.array([3,4,5])
np_c=np_a+np_b
np_c


# In[37]:


# In order to do that, list should be written like this
a=[1,2,3]
b=[3,4,5]
c=[]
for i,j in zip(a,b):
    d=i+j
    c.append(d)
c


# In[40]:


# Another Example
print(np_a/2)
# print(a/2)

# There should be an error with a list


# In Numpy Array, it consists of element and its index.
# 
# <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Images/NumOneList.png" width="660" />

# In[41]:


# Create 1D matrix with specified value
array_1D = np.array([1,2,3])
array_1D


# In[42]:


# An array also can be calculated with addition or other operations
np.array([1,2,3,-1])+1


# <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Images/NumOneAdd.gif" width="500" />

# In[43]:


# Create 2D matrix with specified value
array_2D = np.array([[1,2],[3,4],[5,6]])
array_2D


# In[44]:


# Slicing is an activity to take certain elements from a matrix, use index(from 0)!
print(array_2D[1,1])
print(array_2D[:,1])


# # **Array VS Vector VS Matrix VS Tensor**
# 
# <img src="http://solutionhacker.com/wp-content/uploads/2017/03/python-array.jpg" >

# In[45]:


# Create a matrix with range value
print(np.arange(10))
print(np.arange(3,10))
print(np.arange(3,10,2))


# In[46]:


# how to make a zero matrix with specific size
array_zeros = np.zeros((5,6))
array_zeros


# In[47]:


# how to make a matrix of contents with all values = 1
array_ones = np.ones((5,5))
array_ones


# In[48]:


# how to make identity matrix
array_identity = np.eye((5), dtype=np.str)
array_identity


# ## Numpy - Simple Math

# In[49]:


# Numpy can sum all value inside matrix
np.sum(array_ones)


# In[50]:


# Generate a uniform random number 0-1
array_randomuniform = np.random.rand(2,2)
array_randomuniform


# In[51]:


# Generate a standard normal random number
array_randomnormalstandard = np.random.randn(4,2)
array_randomnormalstandard


# In[52]:


# Generate random integer
array_randomuniform = np.random.randint(2,20,(3,5))
array_randomuniform


# In[53]:


# Generate an interval number
array_lin = np.linspace(0,100,5)
array_lin


# In[54]:


# How to multiply matrix -> Remember linear algebra -> column of matrix 1 = row of matrix 2
a = np.array([[1,2],[3,4]]) 
b = np.array([[11,12],[13,14]]) 
c = np.dot(a,b)
c


# In[55]:


print(np.size(c)) # Return the size or number of elements in matrix
print(np.shape(c)) # Return the shape of matrix


# In[56]:


print(np.pi)
print(np.sin(90))


# <hr>
# <h2>FINISH!</h2>
# <p>Congratulations, you have completed your lesson and hands-on of Numpy in Python. However, there are many things you need to do. You do not need to memorize everything in coding! So, please read here for further explanation <a href="https://numpy.org/doc/stable/" target="_blank">this article</a> to learn how to share your work.
# <hr>

# # 3. Introduction to Pandas
# Pandas (Python for Data Analysis) helps in computing with spreadsheet or tabular data.
# 
# Pandas = Numpy + Relational Database such as SQL.
# 
# There are 2 types of objects in Pandas: 1. Series (1-dimensional) 2. DataFrame (n-dimensional)

# In[57]:


import pandas as pd
#1. Series: Consists of index (0-2) and values
obj=pd.Series([2,3,4],index=[x for x in range(1,4)]) 
obj # index (0-2) and values (1-3). Can be done slice, assign etc


# In[58]:


print(obj.index)
print(obj.values)


# In[59]:


# Series also can be calculated with other series
obj2=pd.Series([4,5,6], index=[1,2,3])
obj+obj2 
# If the index names are not the same it will result in NaN (Not a Number)/NA Not Available


# In[60]:


#2. Dataframe for storing Transaction data per product
data = {
    "roti":[3,2,0,1],
    "cola":[0,3,7,2],
    "tisu":[5,1,1,8],
}
transaction = pd.DataFrame(data)
transaction


# In[61]:


# Add additional index to dataframe
customer = ['Zakka','Adit','Yoga','Asep']
transaction = pd.DataFrame(data, index=customer)
transaction


# In[62]:


transaction.columns


# In[63]:


# SLICING in pandas based on index name
# Take values only in Zakka's transaction
transaction.loc["Zakka"] # It's also possible to use transaction.loc[["Zakka","Yoga"]]


# In[64]:


# SLICING in pandas based on index order
# Take values based on index order
transaction.iloc[2]
# It's also possible to take transaction.iloc[[0,2]]
# It's also possible to take transaction.iloc[0:2]


# In[65]:


# Slicing in pandas based on columns
transaction["roti"]


# In[66]:


# Operations inside Pandas
print(transaction.mean(), end='\n \n')
print(transaction.std(), end='\n \n')
print(transaction.sum(), end='\n \n')
print(transaction.describe(), end='\n \n') #Complete


# Not only that, Pandas is also able to help opening from and saving to other files format such as:
# 
# | Data Format  |        Read       |            Save |
# | ------------ | :---------------: | --------------: |
# | csv          |  `pd.read_csv()`  |   `df.to_csv()` |
# | json         |  `pd.read_json()` |  `df.to_json()` |
# | excel        | `pd.read_excel()` | `df.to_excel()` |
# | hdf          |  `pd.read_hdf()`  |   `df.to_hdf()` |
# | sql          |  `pd.read_sql()`  |   `df.to_sql()` |
# | ...          |        ...        |             ... |
# 

# In[67]:


# For example
import pandas as pd
data_visualization=pd.read_csv('dataset_visualization.csv')
data_visualization
# If we want to take the first 5 rows, use this: data_visualization.head()
# If we want to take the last 5 rows, use this: data_visualization.tail()
# if we want to know total row: len(data_visualization)


# In[68]:


# How many unique data in dataset column?
data_visualization['dataset'].unique()


# In[69]:


# We want to separate each unique data as unique dataframe, for example, just take dino, away, and star data
dino=data_visualization['dataset']== "dino"
dinos=data_visualization[dino]
away=data_visualization["dataset"]=="away"
aways=data_visualization[away]
star=data_visualization["dataset"]=="star"
stars=data_visualization[star]
dinos.head()


# <hr>
# <h2>FINISH!</h2>
# <p>Congratulations, you have completed your lesson and hands-on of Pandas in Python. However, there are many things you need to do. You do not need to memorize everything in coding! So, please read here for further explanation <a href="https://pandas.pydata.org/docs/" target="_blank">this article</a> to learn how to share your work.
# <hr>

# # 4. Matplotlib (Data Visualization)
# Why data visualization? A picture is worth a thousand words. Matplotlob is well-known library in python for visualization. Other library: seaborn, bokeh, plotly, even pandas can visualize.
# 
# There are 2 options in using matplotlib:
# 1. pyplot: does not explicitly specify the customization of the axis, the plot is simpler
# 2. pylab: is a combination of pyplot+numpy used in advanced graphics
# 
# Types of data visualization: 
# 1. Barchart 2. Histogram 3. Pie chart 4. Scatter 5. Line chart 6. Boxplot/Box-and-whisker plot
# 
# There is also so called magic function such as %matplotlib notebook & %matplotlib inline 

# **1. Line Chart**

# In[70]:


# Start from the basic plot
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,10) #Prepare data
plt.plot(x,x, color='green',label='linear')
#Add Legend
plt.legend()
#Show plot
plt.show()


# In[71]:


# Multiple graph
plt.plot([1, 2, 3], [3, 6, 9],linestyle='--')
plt.plot([1, 2, 3], [2, 4, 9],linestyle=':')
plt.show()


# You can embellish the graph by color, types of lines and markers.
# 
# | Char | Color |
# |------|-------|
# |  b   | blue  |
# |  g   | green |
# |  r   | red   |
# |  c   | cyan  |
# |  m   | magenta |
# |  y   | yellow |
# |  k   | black  |
# |  w   | white |
# 
# | Char | Type  |
# |------|-------|
# |  .   | Point |
# |  o   | Circle |
# |  x   | X  |
# |  D   | Diamond |
# |  H   | Hexagon |
# |  s   | Square |
# |  +   | Plus |
# 
# | Char | Style  |
# |------|-------|
# |  -   | Solid |
# |  --   | Dashed |
# |  -.   | Dash-dot |
# |  :   | Dotted |
# |  H   | Hexagon |

# In[72]:


# In the above example, we only 1 plot/image. How if more than 1 plot? PAKE SUPPLOTS
# Basically there are 3 steps: 1. Initialize figure and subfigure/axis 2. plot the data 3. save figure (if required)

# 1. initialize figure
fig=plt.figure(figsize=(6,6)) # Make 1 image with multiple plots Figsize to adjust the size of the image in inches
# initialize subfigure -> It depends with how many plots that we want to visualize
ax1 = fig.add_subplot(121) # add_subplot(xyz): x=the number of rows, y=the number of columns, and z=which section
ax2 = fig.add_subplot(122) 
# So there are 1 row, 2 columns, meaning that there are 2 sections, 1. left & 2. right

# 2. Plot data
ax1.plot([1,2,3,4],[4,3,2,1],label='ax1')
ax1.legend()
ax2.plot([1,2,3,4],[1,2,3,4],label='ax2')
# ax2.legend() -> See what the effect is
plt.show()

# 3. Save data
# plt.savefig('trial.jpg')
# plt.savefig('trial.pdf', transparent=True) # If you want the background to be transparent, not white


# **2. Scatter Chart**
# 
# **Scatter chart can answer the question "How important data visualization is?"**
# 
# Let's see above example in dataset_visualization
# We use 3 data: dinos, aways, and stars. Let's see the statistics of those data. You will notice that those statistics are similar. But is it enough just to believe in statistics only?

# In[73]:


dinos.describe()


# In[74]:


aways.describe()


# In[75]:


stars.describe()


# In[76]:


# Let's use visualize them using scatter plot
dinos.plot.scatter('x', 'y')
aways.plot.scatter('x', 'y')
stars.plot.scatter('x', 'y')
plt.show()


# **3. Pie Chart**

# In[77]:


import matplotlib.pyplot as plt
import numpy as np

labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

data = np.random.rand(7) * 100

plt.pie(data, labels=labels, autopct='%1.1f%%')
plt.axis('equal')
plt.legend()

plt.show()


# **4. Bar Chart**

# In[78]:


import matplotlib.pyplot as plt
import numpy as np

N = 7

x = np.arange(N)
data = np.random.randint(low=0, high=100, size=N)
colors = np.random.rand(N * 3).reshape(N, -1)
labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

plt.title("Weekday Data")
plt.bar(x, data, alpha=0.8, color=colors, tick_label=labels)
plt.show()


# **5. Histogram**

# In[79]:


import matplotlib.pyplot as plt
import numpy as np

data = [np.random.randint(0, n, n) for n in [3000, 4000, 5000]]
labels = ['3K', '4K', '5K']
bins = [0, 100, 500, 1000, 2000, 3000, 4000, 5000]

plt.hist(data, bins=bins, label=labels)
plt.legend()

plt.show()


# Boxplot will be shown in the EDA chapter

# # 5. DATA PREPROCESSING
# 
# The dataset file can be either an URL or your local file address. Previous one, we have called it from local file. Now we will try to open it through URL from UCI database.
# 
# Because the data does not include headers, we can add an argument headers = None inside the read_csv() method, so that pandas will not automatically set the first row as a header.
# You can also assign the dataset to any variable you create.

# In[314]:


# Import pandas library
import pandas as pd
import numpy as np

# Read the online file by the URL provides above, and assign it to variable "df"
other_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(other_path, header=None)
df


# <h3>Add Headers</h3>
# <p>
# Take a look at our dataset; pandas automatically set the header by an integer from 0.
# </p>
# <p>
# To better describe our data we can introduce a header, this information is available at:  <a href="https://archive.ics.uci.edu/ml/datasets/Automobile" target="_blank">https://archive.ics.uci.edu/ml/datasets/Automobile</a>
# </p>
# <p>
# Thus, we have to add headers manually.
# </p>
# <p>
# Firstly, we create a list "headers" that include all column names in order.
# Then, we use <code>dataframe.columns = headers</code> to replace the headers by the list we created.
# </p>
# 

# In[315]:


# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers
df.head()


# **HANDLING MISSING VALUE**
# 
# If we see the second column, it has some missing values indicated with "?". Other datasets may not be indicated with "?", but it will be indicated with NaN (Not a Number), which is Python's default missing value marker. In order to handle Missing Value automatically, we need to replace the "?" symbol with NaN so the dropna() can remove the missing values

# In[316]:


# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head(5)


# <h4>How to count missing values in our dataset</h4>
# 
# The missing values have been converted to Python's default. We use Python's built-in functions to identify these missing values. There are two methods to detect missing data:
# 
# <ol>
#     <li><b>.isnull()</b></li>
#     <li><b>.notnull()</b></li>
# </ol>
# The output is a boolean value indicating whether the value that is passed into the argument is in fact missing data.
# 
# "True" stands for missing value, while "False" stands for not missing value.

# In[317]:


missing_data = df.isnull()
missing_data.head(5)


# In[318]:


# Count missing values in each column
for column in missing_data.columns.values:
    print(column)
    print (missing_data[column].value_counts())
    print("")    


# <h3 id="deal_missing_values">Deal with missing data</h3>
# <b>How to deal with missing data?</b>
# 
# <ol>
#     <li>drop data<br>
#         a. drop the whole row<br>
#         b. drop the whole column
#     </li>
#     <li>replace data<br>
#         a. replace it by mean<br>
#         b. replace it by frequency<br>
#         c. replace it based on other functions
#     </li>
# </ol>
# 
# Whole columns should be dropped only if most entries in the column are empty. In our dataset, none of the columns are empty enough to drop entirely. We have some freedom in choosing which method to replace data; however, some methods may seem more reasonable than others. We will apply each method to many different columns:
# 
# <b>Replace by mean:</b>
# 
# <ul>
#     <li>"normalized-losses": 41 missing data, replace them with mean</li>
#     <li>"stroke": 4 missing data, replace them with mean</li>
#     <li>"bore": 4 missing data, replace them with mean</li>
#     <li>"horsepower": 2 missing data, replace them with mean</li>
#     <li>"peak-rpm": 2 missing data, replace them with mean</li>
# </ul>
# 
# <b>Replace by frequency:</b>
# 
# <ul>
#     <li>"num-of-doors": 2 missing data, replace them with "four". 
#         <ul>
#             <li>Reason: 84% sedans is four doors. Since four doors is most frequent, it is most likely to occur</li>
#         </ul>
#     </li>
# </ul>
# 
# <b>Drop the whole row:</b>
# 
# <ul>
#     <li>"price": 4 missing data, simply delete the whole row
#         <ul>
#             <li>Reason: price is what we want to predict. Any data entry without price data cannot be used for prediction; therefore any row now without price data is not useful to us</li>
#         </ul>
#     </li>
# </ul>
# 

# <h4>Calculate the average of the column </h4>

# In[319]:


avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)


# <h4>Replace "NaN" by mean value in "normalized-losses" column</h4>
# 

# In[320]:


df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)


# <h4>Do the same things to other columns -> Replace NaN by Mean Value</h4>
# 

# In[321]:


avg_bore=df['bore'].astype('float').mean(axis=0)
df["bore"].replace(np.nan, avg_bore, inplace=True)
strokemean = df['stroke'].astype('float').mean()
df['stroke'].replace(np.nan,strokemean,inplace=True)
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)


# <h4>Replace "NaN" by frequency value in "num-of-doors" column</h4>
# To see which values are present in a particular column, we can use the ".value_counts()" method. We can see that four doors are the most common type. We can also use the ".idxmax()" method to calculate for us the most common type automatically: df['num-of-doors'].value_counts().idxmax()

# In[322]:


df['num-of-doors'].value_counts()


# The replacement procedure is very similar to what we have seen previously

# In[323]:


#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)
# We also can use: df.fillna("four",inplace=True)


# <h4>Finally, let's drop all rows that do not have price data:</h4>

# In[324]:


# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)

df.head()


# <b>Good!</b> Now, we obtain the dataset with no missing values.
# 
# <h3 id="correct_data_format">Correct data format</h3>
# <b>We are almost there!</b>
# <p>The last step in data cleaning is checking and making sure that all data is in the correct format (int, float, text or other).</p>
# 
# In Pandas, we use 
# 
# <p><b>.dtype()</b> to check the data type</p>
# <p><b>.astype()</b> to change the data type</p>
# 
# 

# In[325]:


df.dtypes


# <p>As we can see above, some columns are not of the correct data type. Numerical variables should have type 'float' or 'int', and variables with strings such as categories should have type 'object'. For example, 'normalized-losses', 'bore', and 'stroke' variables are numerical values, so we should expect them to be of the type 'float' or 'int'; however, they are shown as type 'object'. We have to convert data types into a proper format for each column using the "astype()" method.</p> 
# 
# <h4>Convert data types to proper format</h4>
# 

# In[326]:


df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df.dtypes


# <h2 id="Data Transformation">Data Transformation</h2>
# <p>
# Data is usually collected from different agencies with different formats. Data Transformation is the process of transforming data into a common format which allows the researcher to make the meaningful comparison.
# </p>
# 
# <b>Example</b>
# 
# <p>Transform mpg to L/100km:</p>
# <p>In our dataset, the fuel consumption columns "city-mpg" and "highway-mpg" are represented by mpg (miles per gallon) unit. Assume we are developing an application in Indonesia that accept the fuel consumption with L/100km standard</p>
# <p>We will need to apply <b>data transformation</b> to transform mpg into L/100km?</p>
# 

# <p>The formula for unit conversion is<p>
# L/100km = 235 / mpg
# <p>We can do many mathematical operations directly in Pandas.</p>
# 

# In[327]:


df.head()


# In[328]:


# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]
df["highway-L/100km"]=235/df["highway-mpg"]
# check your transformed data 
# pd.set_option("display.max_rows",None, "display.max_columns", None) # If we want to show all dataset
df


# <h2 id="Feature Scaling">Feature Scaling</h2>
# 
# <b>Why Feature Scaling? Here we use Normalization</b>
# 
# <p>Normalization is the process of transforming values of several variables into a similar range. Typical normalizations include scaling the variable so the variable average is 0, scaling the variable so the SD is 1, or scaling variable so the variable values range from 0 to 1
# </p>
# 
# <b>Example</b>
# 
# <p>To demonstrate normalization, let's say we want to scale the columns "length", "width" and "height" </p>
# <p><b>Target:</b>would like to Normalize those variables so their value ranges from 0 to 1.</p>
# <p><b>Approach:</b> replace original value by (original value)/(maximum value) -> Here we use the simplest approach of normalization. Note that in PPT, the approach is different</p>
# 

# In[329]:


# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df["height"]=df["height"]/df["height"].max()
# Other Approaches
# df["height"]=(df["height"]-df["height"].min())/(df["height"].max()-df["height"].min()) # Normalization
# df["height"]=(df["height"]-df["height"].mean())/(df["height"].std()) # Standardization


# In[330]:


df[["length","width","height"]].head()


# <h2 id="binning">Binning/Discretization</h2>
# <b>What is binning?</b>
# <p>
#     Binning is a process of transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis.
# </p>
# 
# <b>Example: </b>
# 
# <p>In our dataset, "horsepower" is a real valued variable ranging from 48 to 288, it has 57 unique values. What if we only care about the price difference between cars with high horsepower, medium horsepower, and little horsepower (3 types)? Can we rearrange them into three ‘bins' to simplify analysis? </p>
# 
# <p>We will use the Pandas method 'cut' to segment the 'horsepower' column into 3 bins </p>
# 

# <h3>Example of Binning Data In Pandas</h3>
# 

#  Convert data to correct format 
# 

# In[331]:


df["horsepower"]=df["horsepower"].astype(int, copy=True)


# Lets plot the histogram of horspower, to see what the distribution of horsepower looks like.
# 

# In[332]:


import matplotlib.pyplot as plt
plt.hist(df["horsepower"])

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")


# <p>We would like 3 bins of equal size bandwidth so we use numpy's <code>linspace(start_value, end_value, numbers_generated)</code> function.</p>
# <p>Since we want to include the minimum value of horsepower we want to set start_value=min(df["horsepower"]).</p>
# <p>Since we want to include the maximum value of horsepower we want to set end_value=max(df["horsepower"]).</p>
# <p>Since we are building 3 bins of equal length, there should be 4 dividers, so numbers_generated=4.</p>
# 

# We build a bin array, with a minimum value to a maximum value, with bandwidth calculated above. The bins will be values used to determine when one bin ends and another begins.
# 

# In[333]:


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


#  We set group  names:
# 

# In[334]:


group_names = ['Low', 'Medium', 'High']


#  We apply the function "cut" the determine what each value of "df['horsepower']" belongs to. 
# 

# In[335]:


df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)
df[['horsepower','horsepower-binned']].head(50)


# Lets see the number of vehicles in each bin.
# 

# In[336]:


df["horsepower-binned"].value_counts()


# Lets plot the distribution of each bin.
# 

# In[337]:


import matplotlib.pyplot as plt
plt.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")


# <p>
#     Check the dataframe above carefully, you will find the last column provides the bins for "horsepower" with 3 categories ("Low","Medium" and "High"). 
# </p>
# <p>
#     We successfully narrow the intervals from 57 to 3!
# </p>
# 

# # Exploratory Data Analysis
# After you get the clean data and make the format based on what you want. Sometimes you need to do EDA to understand more about your data. Some important things you can do in EDA are:
# 1. Correlation
# 2. Descriptive Statistical Analysis
# 3. ANOVA

# # 1. Correlation
# 
# Remember Correlation is not Causation
# - Correlation: a measure of the extent of interdependence between variables.
# - Causation: the relationship between cause and effect between two variables.
# 
# It is important to know the difference between these two and that correlation does not imply causation. Determining correlation is much simpler the determining causation as causation may require independent experimentation. Correlation helps in finding multicollinearity and duplication of data.
# 
# Here we will see, how to use correlation in EDA. It helps us to find which features should be included for supervised purpose. Basically, there are 2 ways to utilize correlation. First, find correlation between feature and target. Second, find correlation between feature with another feature (handling multicollinearity). Here, we only cover the first utilization. Since you can do the first utilization, easily you can do the second utilization by yourself. We can start from numerical variables and then categorical variables.
# <h2>Numerical variables:</h2> 
# 
# <p>Continuous numerical variables are variables that may contain any value within some range. Continuous numerical variables can have the type "int64" or "float64". A great way to visualize these variables is by using scatterplots with fitted lines.</p>
# 
# <p>In order to start understanding the (linear) relationship between an individual variable and the price. We can do this by using "regplot", which plots the scatterplot plus the fitted regression line for the data.</p>

#  Let's see several examples of different linear relationships:

# <h3>Positive linear relationship</h3>

# Let's find the scatterplot of "engine-size" and "price" 

# In[338]:


import seaborn as sns
# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)


# <p>As the engine-size goes up, the price goes up: this indicates a positive direct correlation between these two variables. Engine size seems like a pretty good predictor of price since the regression line is almost a perfect diagonal line.</p>

#  We can examine the correlation between 'engine-size' and 'price' and see it's approximately  0.87

# In[339]:


df[["engine-size", "price"]].corr()


# Therefore, engine size is a potential predictor variable of price since the correlation is more than 0.5

# <h3>Negative linear relationship</h3>

# Let's find the scatterplot of "Highway mpg" and "price" 

# In[340]:


sns.regplot(x="highway-mpg", y="price", data=df)


# <p>As the highway-mpg goes up, the price goes down: this indicates an inverse/negative relationship between these two variables. Highway mpg could potentially be a predictor of price.</p>

# We can examine the correlation between 'highway-mpg' and 'price' and see it's approximately  -0.704

# In[341]:


df[['highway-mpg', 'price']].corr()


# Therefore, highway-mpg is a potential predictor variable of price since the correlation is less than -0.5

# <h3>Weak Linear Relationship</h3>

# Let's see if "Peak-rpm" as a predictor variable of "price".

# In[342]:


sns.regplot(x="peak-rpm", y="price", data=df)


# <p>Peak rpm does not seem like a good predictor of the price at all since the regression line is close to horizontal. Also, the data points are very scattered and far from the fitted line, showing lots of variability. Therefore it's it is not a reliable variable.</p>

# We can examine the correlation between 'peak-rpm' and 'price' and see it's approximately -0.101616 

# In[343]:


df[['peak-rpm','price']].corr()


#  sometimes we would like to know the significant of the correlation estimate. 

# <b>P-value</b>: 
# <p>What is this P-value? The P-value is the probability value that the correlation between these two variables is statistically significant. Normally, we choose a significance level of 0.05, which means that we are 95% confident that the correlation between the variables is significant.</p>
# 
# By convention, when the
# <ul>
#     <li>p-value is $<$ 0.001: we say there is a very strong evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.05: there is strong evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.1: there is weak evidence that the correlation is significant.</li>
#     <li>the p-value is $>$ 0.1: there is no evidence that the correlation is significant.</li>
# </ul>

#  We can obtain this information using  "stats" module in the "scipy"  library.

# For example, let's calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'. 

# In[344]:


from scipy import stats
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# <h5>Conclusion:</h5>
# <p>Since the p-value is $<$ 0.001, the correlation between wheel-base and price is statistically significant, although the linear relationship isn't extremely strong (~0.585)</p>

# <h3>Categorical variables</h3>
# 
# <p>These are variables that describe a 'characteristic' of a data unit, and are selected from a small group of categories. The categorical variables can have the type "object" or "int64". A good way to visualize categorical variables is by using boxplots or box-and-whisker plot.</p>

# Let's look at the relationship between "body-style" and "price".

# In[345]:


sns.boxplot(x="body-style", y="price", data=df) # Add showmeans=True argument if we want to see the mean


# <p>We see that the distributions of price between the different body-style categories have a significant overlap, and so body-style would not be a good predictor of price. Let's examine engine "engine-location" and "price":</p>

# In[346]:


sns.boxplot(x="engine-location", y="price", data=df)


# <p>Here we see that the distribution of price between these two engine-location categories, front and rear, are distinct enough to take engine-location as a potential good predictor of price.</p>

# In[347]:


sns.boxplot(x="drive-wheels", y="price", data=df)


# Here we see that the distribution of price between the different drive-wheels categories differs; as such drive-wheels could potentially be a predictor of price.

# <h2 id="discriptive_statistics">2. Descriptive Statistical Analysis</h2>

# <p>Let's first take a look at the variables by utilizing a description method.</p>
# 
# <p>The <b>describe</b> function automatically computes basic statistics for all continuous variables. Any NaN values are automatically skipped in these statistics.</p>
# 
# This will show:
# <ul>
#     <li>the count of that variable</li>
#     <li>the mean</li>
#     <li>the standard deviation (std)</li> 
#     <li>the minimum value</li>
#     <li>the IQR (Interquartile Range: 25%, 50% and 75%)</li>
#     <li>the maximum value</li>
# <ul>
# 

#  We can apply the method "describe" as follows:

# In[348]:


df.describe()


#  The default setting of "describe" skips variables of type object. We can apply the method "describe" on the variables of type 'object' as follows:

# In[349]:


df.describe(include=['object'])


# <h3>Value Counts for Detecting Imbalance Data</h3>

# <p>Value-counts is a good way of understanding how many units of each categorical variable we have. We can apply the "engine-location" method on the column 'drive-wheels'. Don’t forget the method "value_counts" only works on Pandas series, not Pandas Dataframes. As a result, we only include one bracket "df['drive-wheels']" not two brackets "df[['drive-wheels']]".</p>

# In[350]:


# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts


# <p>Examining the value counts of the engine location would not be a good predictor variable for the price. This is because we only have three cars with a rear engine and 198 with an engine in the front, this result is skewed. Thus, we are not able to draw any conclusions about the engine location. Try other variables by yourself</p>

# <h2 id="ANOVA">3. ANOVA</h2>

# <p>The Analysis of Variance  (ANOVA) is a statistical method used to test whether there are significant differences between the means of three or more groups. ANOVA returns two parameters:</p>
# 
# <p><b>F-test score</b>: ANOVA assumes the means of all groups are the same, calculates how much the actual means deviate from the assumption, and reports it as the F-test score. A larger score means there is a larger difference between the means.</p>
# 
# <p><b>P-value</b>:  P-value tells how statistically significant is our calculated score value.</p>
# 
# <p>If our price variable is strongly correlated with the variable we are analyzing, expect ANOVA to return a sizeable F-test score and a small p-value.</p>

# In[351]:


# grouping using groupby
df_group=df.groupby(['drive-wheels'])
df_group.first()


# In[352]:


df_group.get_group('4wd')['price']


# we can use the function 'f_oneway' in the module 'stats' to obtain the F-test score and P-value.

# In[353]:


from scipy import stats
# ANOVA
f_val, p_val = stats.f_oneway(df_group.get_group('fwd')['price'], df_group.get_group('rwd')['price'], df_group.get_group('4wd')['price'])  
#  stats.f_oneway(sample1,sample2,...sample n)
print( "ANOVA results: F=", f_val, ", P =", p_val)   


# This is a great result, with a large F test score and a P value of almost 0 implying that each category is significantly different based on the price value. To get more detail information, you can test the mean difference for each category such as fwd vs rwd, rwd vs 4wd, and fwd vs 4wd.

# <h3>Conclusion: From EDA, we will know the important variables</h3>

# <p>We now have a better idea of what our data looks like and which variables are important to take into account when predicting the car price. We have narrowed it down to the following variables:</p>
# 
# Continuous numerical variables:
# <ul>
#     <li>Length</li>
#     <li>Width</li>
#     <li>Curb-weight</li>
#     <li>Engine-size</li>
#     <li>Horsepower</li>
#     <li>City-mpg</li>
#     <li>Highway-mpg</li>
#     <li>Wheel-base</li>
#     <li>Bore</li>
# </ul>
#     
# Categorical variables:
# <ul>
#     <li>Drive-wheels</li>
# </ul>
# 
# <p>As we now move into building machine learning models to automate our analysis, feeding the model with variables that meaningfully affect our target variable will improve our model's prediction performance.</p>

# ## Congratulations You Have Finished A Big Step!
# 
# I expect you from now, you have introduced the python programming language as well as how to use it for data pre-processing!
# Those knowledge will be useful not only for data science purpose but for general purpose such as for doing optimization, simulation, until creating application!
# 
# Actually, there are many ways in conducting programming. Previous material is not the most efficient way, because I want you to be more experienced especially for the logic behind it! We will see the more efficient way in the next course!

# # 6. REGRESSION

# <h4>Linear Regression</h4>

# 
# <p>One example of a Data  Model that we will be using is</p>
# <b>Simple Linear Regression</b>.
# 
# <br>
# <p>Simple Linear Regression is a method to help us understand the relationship between two variables:</p>
# <ul>
#     <li>The predictor/independent variable (X)</li>
#     <li>The response/dependent variable (that we want to predict)(Y)</li>
# </ul>
# 
# <p>The result of Linear Regression is a <b>linear function</b> that predicts the response (dependent) variable as a function of the predictor (independent) variable.</p>
# 
# 

# $$
#  Y: Response \ Variable\\
#  X: Predictor \ Variables
# $$
# 

#  <b>Linear function:</b>
# $$
# Yhat = a + b  X
# $$

# <ul>
#     <li>a refers to the <b>intercept</b> of the regression line0, in other words: the value of Y when X is 0</li>
#     <li>b refers to the <b>slope/coefficient</b> of the regression line, in other words: the value with which Y changes when X increases by 1 unit</li>
# </ul>
# 
# **Use 5 approaches in predictive modeling:**
# 1. Import library & dataset as well as preprocessing the data (feature scaling and dummy variable)
# 2. Split Data into Training & Testing data. In regression part, we will split the data directly.
# 3. Fitting the model to Training data
# 4. Predict Testing data
# 5. Evaluate the result (visualization & metrics)

# <h4>No, assume we want to know: How could Highway-mpg help us predict car price?</h4>

# For this example, we want to look at how highway-mpg can help us predict car price.
# Using simple linear regression, we will create a linear function with "highway-mpg" as the predictor variable and the "price" as the response variable.
# 
# Note: we will use train test split directly here for all data (X and y).

# In[354]:


df.head()


# In[355]:


# Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
X = df.iloc[:,24:25].values # df.columns.get_loc("highway-mpg") 
y = df.iloc[:,25].values # df.columns.get_loc("price")

# Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Fitting Simple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict Testing data
y_pred = regressor.predict(X_test)
print("Coefficient: ", regressor.coef_)
print("Intercept: ", regressor.intercept_)


# In[121]:


# Visualize training data
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('highway-mpg vs price (Training set)')
plt.xlabel('highway-mpg')
plt.ylabel('price')
plt.show()
 
# Visualize testing data
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('highway-mpg vs price (Testing set)')
plt.xlabel('highway-mpg')
plt.ylabel('price')
plt.show()

# Evaluate Metrics
from sklearn import metrics
print("MAE =",metrics.mean_absolute_error(y_test,y_pred))
print("MSE =",metrics.mean_squared_error(y_test,y_pred))
print("R2 =",metrics.r2_score(y_test,y_pred))


# In[122]:


plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()


# In[123]:


# If we want to predict a specified value
regressor.predict(np.array(23).reshape(1,-1))


# <h4>Multiple Linear Regression</h4>

# <p>What if we want to predict car price using more than one variable?</p>
# 
# <p>If we want to use more variables in our model to predict car price, we can use <b>Multiple Linear Regression</b>.
# Multiple Linear Regression is very similar to Simple Linear Regression, but this method is used to explain the relationship between one continuous response (dependent) variable and <b>two or more</b> predictor (independent) variables.
# Most of the real-world regression models involve multiple predictors.</p>
# 
# <p>From the previous section  we know that other good predictors of price could be:</p>
# <ul>
#     <li>engine-size</li>
#     <li>highway-mpg</li>
#     <li>drive-wheels</li>
# </ul>
# Remember, drive-wheels is a variable containing 3 unique categorical variables (n) (4wd, fwd, rwd). Therefore, we need to use dummy variable where the number of dummy columns are n_new=n-1=3-1=2. So drive-wheels will be converted with 2 new variables.
# Let's develop a model using these variables as the predictor variables.

# $$
# Y: Response \ Variable\\
# X_1 :Predictor\ Variable \ 1\\
# X_2: Predictor\ Variable \ 2\\
# X_3: Predictor\ Variable \ 3\\
# X_4: Predictor\ Variable \ 4\\
# $$

# $$
# a: intercept\\
# b_1 :coefficients \ of\ Variable \ 1\\
# b_2: coefficients \ of\ Variable \ 2\\
# b_3: coefficients \ of\ Variable \ 3\\
# b_4: coefficients \ of\ Variable \ 4\\
# $$

# The equation is given by

# $$
# Yhat = a + b_1 X_1 + b_2 X_2 + b_3 X_3 + b_4 X_4
# $$

# The first 2 X describes engine-size and highway-mpg, and the last 2 X describes dummy variables of drive-wheels.

# In[356]:


# Let's take the data from df using loc function
df.loc[:,['drive-wheels','engine-size','highway-mpg','price']]


# In[366]:


# Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
X_regression = df.loc[:,['drive-wheels','engine-size','highway-mpg']].values
y_regression = df.iloc[:,25].values # df.columns.get_loc("price")

X_regression


# In[367]:


# Encode categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(
        [('encoder', OneHotEncoder(), [0])], # Which column contains categorical variable
        remainder='passthrough')
X_regression = np.array(transformer.fit_transform(X_regression), dtype=np.float)
# Run from dummy variabel trap
X_regression = X_regression[:, 1:] # We discard the first column
X_regression


# In[368]:


# Be aware that the scala of each independent variables are different
# Let's do feature scaling using MinMaxScaler()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # another one is StandardScaler()
# transform data
X_regression = scaler.fit_transform(X_regression)
X_regression


# Note: here we scale all data directly using **fit_transform(X_regression)** for simplicity. Some researchers may argue that this approach will cause data leakage. Therefore, later you can split the data first then do feature scaling using this approach:
# 
# <code>scaler = preprocessing.StandardScaler().fit(X_train)
# X_train_transformed = scaler.transform(X_train)
# clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
# X_test_transformed = scaler.transform(X_test)<code>

# In[128]:


# Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size = 0.3, random_state = 0)

# Fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict Testing data
y_pred = regressor.predict(X_test)
print("Coefficient: ", regressor.coef_)
print("Intercept: ", regressor.intercept_)


# In[129]:


# Because there are too many variables, we do not need to visualize it, we can directly use metrics evaluation
# Evaluate Metrics
from sklearn import metrics
print("MAE =",metrics.mean_absolute_error(y_test,y_pred))
print("MSE =",metrics.mean_squared_error(y_test,y_pred))
print("R2 =",metrics.r2_score(y_test,y_pred))


# In[130]:


plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()


# <h4>Polynomial Regression</h4>

# <p><b>Polynomial regression</b> is a particular case of the general linear regression model or multiple linear regression models.</p> 
# <p>We get non-linear relationships by squaring or setting higher-order terms of the predictor variables.</p>
# 
# <p>There are different orders of polynomial regression (MLR only use 1st order):</p>

# <center><b>Quadratic - 2nd order</b></center>
# $$
# Yhat = a + b_1 X +b_2 X^2 
# $$
# 
# 
# <center><b>Cubic - 3rd order</b></center>
# $$
# Yhat = a + b_1 X +b_2 X^2 +b_3 X^3\\
# $$
# 
# 
# <center><b>Higher order</b>:</center>
# $$
# Y = a + b_1 X +b_2 X^2 +b_3 X^3 + ....+ b_n X^n\\
# $$
# If we see from the graphic, the difference between polynomial with simpel linear regression is as follows:
# <img src = "https://miro.medium.com/max/1400/1*zOl_ztYqnzyWRkBffeOsRQ.png" align = "center" width="460" />
# Basically we can use polynomial by providing more feature first based on the order, then we can process it using usual linear legression model.
# <img src = "https://cdn.analyticsvidhya.com/wp-content/uploads/2020/03/pr7.png" align = "center" width="460" />
# 
# See the example below:
# ```
# >>> X = np.arange(6).reshape(3, 2)
# >>> X
# array([[0, 1],
#        [2, 3],
#        [4, 5]])
# >>> poly = PolynomialFeatures(2)
# >>> poly.fit_transform(X)
# array([[ 1.,  0.,  1.,  0.,  0.,  1.],
#        [ 1.,  2.,  3.,  4.,  6.,  9.],
#        [ 1.,  4.,  5., 16., 20., 25.]])
# ```

# <p>Let's use this approach by using the same dataset as previous one</p>

# In[131]:


# Fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree = 2)  # We can change this degree
X_poly = poly_reg2.fit_transform(X_regression)
X_poly


# In[132]:


# Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y_regression, test_size = 0.3, random_state = 0)

# Fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict Testing data
y_pred = regressor.predict(X_test)
print("Coefficient: ", regressor.coef_)
print("Intercept: ", regressor.intercept_)


# In[133]:


# Because there are too many variables, we do not need to visualize it, we can directly use metrics evaluation
# Evaluate Metrics
from sklearn import metrics
print("MAE =",metrics.mean_absolute_error(y_test,y_pred))
print("MSE =",metrics.mean_squared_error(y_test,y_pred))
print("R2 =",metrics.r2_score(y_test,y_pred))


# In[134]:


plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()


# You can compare the result with the MLR result, and you can try to change the order of polynomial regression!

# # 7. CLASSIFICATION

# In this classification and the next clustering problem, we will use the well-known dataset so-called Iris Dataset. Iris has three flower types, and we want to predict the flower type based on length and width of sepal and petal. See the figure below:
# <img src = "https://machinelearninghd.com/wp-content/uploads/2021/03/iris-dataset.png" align = "center" width="490" />

# In[135]:


# Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Import dataset
df1=pd.read_csv("dataset_classification_clustering_iris.csv")
df1


# **Data Pre-Processing**:
# From dataset above, we need to do feature scaling and encoding the class into number
# 
# **Note**:
# We have 4 features with 3 classes in target. We also have 150 clean data.

# In[136]:


# Feature Scaling using Normalization
from sklearn.preprocessing import MinMaxScaler
X=MinMaxScaler().fit_transform(df1.iloc[:,:4])
X


# In[137]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df1["class"])
y


# **Now, data are ready. We will try to use 6 different algorithms and evaluate them using cross-validation. The algorithms are:**
# 1. Support Vector Classification (SVC)
# 2. Naive Bayes (NB)
# 3. Decision-Tree (DT)
# 4. Random Forest (RF)
# 5. K-Nearest Neighbor (KNN)
# 6. Logistic Regression (LR)

# ### SUPPORT VECTOR CLASSIFICATION (SVC)
# SVC is one of the techniques in Support Vector Machine (SVM) model. In simple way, SVC is basically trying to find a **hyperplane** by machine to support splitting the data so that it can be classified. The objective is finding the best hyperplane so that it can maximize the margin. Margin itself is the perpendicular distance of the support vectors (outermost point) for each class. Let's see the illustration below.
# 
# <img src = "https://www.researchgate.net/publication/331308937/figure/fig1/AS:870140602249216@1584469094771/Illustration-of-support-vector-machine-SVM-to-generalize-the-optimal-separating.ppm" align = "center" width="460" />

# First, let me give you example how to use **split data into training and testing** for SVC model:

# In[138]:


# Split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# Fit the model to training data
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
 
# Test the model to testing data
y_pred_training = classifier.predict(X_train)
 
# Evaluate it using Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred_training)
print('Confusion Matrix: \n',cm)

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train, y_pred_training)
print('accuracy for training=', accuracy)


# In[139]:


# Test the model to testing data
y_pred = classifier.predict(X_test)
 
# Evaluate it using Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: \n',cm)

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print('accuracy for testing =', accuracy)


# In[140]:


# How to know the detail of the model? Not all model can provide this. Try SVC with another karnel such as: rbf Why?
print('classes: ',classifier.classes_)
print('coefficients: ',classifier.coef_)
print('intercept :', classifier.intercept_)


# In[141]:


# How if we want to predict selected value? for example new_data=[0.2,0.3,0.1,0.4]
classifier.predict(np.array([[0.2,0.3,0.1,0.4]]))
# Can you guess what the result means?


# Second, let's see if we use **cross-validation (CV)**. CV is usually needed when we want to optimize the model either selecting feature, hyperparameter-tuning, etc. But here, I only show you how to use it in general without any feature selection. Later, we will discuss how to use CV for optimization. This approach can be computationally expensive, but does not waste too much data due to many splits.

# In[142]:


# Create SVC model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
# Split the data using CV with K=5
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
# Evaluate the accuracy
scores.mean()


# In[143]:


# If we want to see the result of classification
from sklearn.model_selection import cross_val_predict
cross_val_predict(classifier, X, y, cv=5)


# Here you see that we can easily implement cross validation in sklearn. Now, how to use CV for regression? we just need to change the scoring parameter into 'neg_mean_squared_error'. Why negative? Because Sklearn community has made convention that all scoring in CV should be higher then better. Therefore, we need to make negative value. For full metrics, follow this: https://scikit-learn.org/stable/modules/model_evaluation.html

# ### NAIVE BAYES
# Naive Bayes or sometimes called as Naive Bayes Classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem. it is called Naive because this model ignores an assumption in the theorem which is variable independent. This is totally just a formula that is shown below:
# <img src = "https://miro.medium.com/max/1200/0*Z3nK2E6TghNWzZGz.png" align = "center" width="460" />

# In[144]:


# Create Naive Bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
# Split the data using CV with K=5
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
# Evaluate the accuracy
scores.mean()


# ### DECISION TREE
# This model is one of techniques in tree-based model. This model divides the data into several groups gradually. The division starts from the first decision. then the results of the first decision are used to make the second decision. The results of the first and second decisions are used to make the third decision, and so on. This division is assessed based on some criteria such as gini index and entropy.
# <img src = "http://mines.humanoriented.com/classes/2010/fall/csci568/portfolio_exports/lguo/image/decisionTree/decisionTree.jpg" align = "center" width="460" />

# In[145]:


# Create Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# Split the data using CV with K=5
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
# Evaluate the accuracy
scores.mean()


# ### RANDOM FOREST
# This model is one of techniques in tree-based model. The principle of this model is similar with decision tree. Therefore, sometimes this model is also called as an extension of Decision Tree. Random Forest will try to create so many tries (n). Each tree will predict based on their decision. From all trees, then we can take the best one based on majority voting.
# <img src = "https://static.javatpoint.com/tutorial/machine-learning/images/random-forest-algorithm2.png" align = "center" width="460" />

# In[146]:


# Create Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
# Split the data using CV with K=5
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
# Evaluate the accuracy
scores.mean()


# ### K-NEAREST NEIGHBOR (KNN)
# KNN is a simple technique for classification. From its name, this method tries to find the K (number of) nearest neighbor from new data based on euclidean distance. Then count which class is the most included and classify the new data to them. See the illustration below.
# 
# <img src = "https://miro.medium.com/max/405/0*rc5_e6-6AHzqppcr" align = "center" width="460" />

# In[147]:


# Create KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
# Split the data using CV with K=5
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
# Evaluate the accuracy
scores.mean()


# ### LOGISTIC REGRESSION
# In contrast to regression in general, logistic regression is a regression technique whose function is to separate the dataset into two parts (groups). This function is a combination of simple linear regression + Sigmoid function. Therefore, it is a linear classification technique. Basically, Logistic regression is only able to answer binary problem which is in the form of Yes (1)/No (0) such as Infected/Not Infected. But, now there is approach to handle multi-class problem so-clled multi-nomial logistic regression which is classifying based on probability theory. This model tries to fit the S line so that it can divide the class based on the probability. See the illustration below.
# 
# <img src = "https://rajputhimanshu.files.wordpress.com/2018/03/linear_vs_logistic_regression.jpg" align = "center" width="480" />

# In[148]:


# Create Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) # multi-class is auto
# Split the data using CV with K=5
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
# Evaluate the accuracy
scores.mean()


# After you try all algorithms and evaluate them with cross validation, then choose the best model. After that, fit all the data (retrain) for selected algorithm and use it to predict new data later. By doing so, when you face small dataset, you still can use all data for training.

# ## Congratulations You Have Finished A Big Step!
# 
# I expect you from now, you have introduced predictive analytics consisting of regression and classification!
# 
# Actually, there are many algorithms outhere. There is no the best algorithm! Sometimes, good algorithm is only good for specific dataset. Therefore, the best way is to try as many algorithms as you can, then choose the most accurate one. After we finish ML part, we will learn how to use deep learning as advanced technique to make a great prediction!
# 
# After this, we will deep dive into descriptive analytics! Stay hungry!

# # 8. CLUSTERING

# First, remember the dataset that we will use is still the same, Iris Dataset. Supposed that we do not know the target of each class. So we only have X data, not until y data. Can we make clustering algorithm that can specify the class accurately?
# 
# Remember to always scale the feature. Why we need to scale for clustering? Because clustering use distance as measurementFirst, remember the dataset that we will use is still the same, Iris Dataset. Supposed that we do not know the target of each class. So we only have X data, not until y data. Can we make clustering algorithm that can specify the class accurately?
# 
# Use 5 steps in clustering modeling:
# 
# 1. Import library & dataset as well as preprocessing the data
# 2. Fitting the model to the data
# 3. Cluster the data
# 4. Evaluate the result (visualization & metrics)

# In[149]:


# We can directly recall the result of pre-processing data
X


# **K-Means Clustering**
# 

# In[150]:


# Import library
from sklearn.cluster import KMeans
# Fit the model to the data 
# Suppose we want to cluster data into n_clusters = 2 clusters
kmeans = KMeans(n_clusters = 2, random_state = 2)
kmeans.fit(X)
print("clustering result:", kmeans.labels_)


# In[151]:


# Evaluate with the metric: SSE should be minimum!
print("SSE:",kmeans.inertia_)
print("Centroids of data: \n",kmeans.cluster_centers_)
# Evaluate with visualization
import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.scatter(X[:,2], X[:,3], c=kmeans.labels_) # Visualize the cluster result
ax.set_title('K-Means Clustering Results with K=2')
ax.scatter(kmeans.cluster_centers_[:,2], kmeans.cluster_centers_[:,3], marker='+', s=100) # Visualize the centroid result
plt.show()


# In[152]:


# Using the elbow method to determine the number of clusters
sse = []
# Let's say we want to evaluate K from 1 to 11
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 2)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
plt.plot(range(1, 11), sse)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()


# **Let's discuss, what is the appripriate number of clusters based on elbow method?**

# In[153]:


# Try with 3 clusters
kmeans = KMeans(n_clusters = 3, random_state = 2)
kmeans.fit(X)
print("clustering result:", kmeans.labels_)


# In[154]:


# Evaluate with the metric: SSE should be minimum!
print("SSE:",kmeans.inertia_)
print("Centroids of data: \n",kmeans.cluster_centers_)
# Evaluate with visualization
import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.scatter(X[:,2], X[:,3], c=kmeans.labels_) # Visualize the cluster result
ax.set_title('K-Means Clustering Results with K=3')
ax.scatter(kmeans.cluster_centers_[:,2], kmeans.cluster_centers_[:,3], marker='+', s=100) # Visualize the centroid result
plt.show()


# In[155]:


# Since we use supervised data, we can measure how good clustering algorithm based on its accuracy
misclassification=0
for i in range(len(y)):
    if y[i] != kmeans.labels_[i]:
        misclassification+=1
print('misclassification rate of K-means =', 100*misclassification/len(y),'%')


# **Hierarchical Clustering**
# 
# From theory that I presented in PPT, you know that there are agglomerative & divisive hierarchical clustering. Those two are only paradigms whether we want to use bottom-up approach or top-down approach, not algorithms. The steps are quiet different with K-means, here we need to create dendogram first, then determine the threshold, and finally determine the clustering result.

# In[156]:


# Import library
from scipy.cluster import hierarchy
# Fit the model to the data 
# We do not need to determine n_clusters first like in K-means
linkage=hierarchy.complete(X) #create hierarchical cluster using complete linkage, other linkages: average, single, ward
f,ax=plt.subplots(1,1,figsize=(20,13)) # prepare dendogram to determine number of cluster based on threshold
plt.xlabel('Data')
plt.ylabel('Euclidean Distance')
cluster=hierarchy.dendrogram(linkage,leaf_font_size=8,ax=ax)


# In[157]:


# Supposed we want to cut in 1, therefore there will be 3 clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
cluster.fit_predict(X)
print("clustering result:", cluster.labels_)


# In[158]:


# In Hierarchical clustering, it does not calculate the SSE and the centroids, therefore we are only able to visualize it
# Evaluate with visualization
import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.scatter(X[:,2], X[:,3], c=cluster.labels_) # Visualize the cluster result
ax.set_title('Hierarchical Clustering with Complete Linkage')
plt.show()


# In[159]:


# Since we use supervised data, we can measure how good clustering algorithm based on its accuracy
misclassification=0
for i in range(len(y)):
    if y[i] != cluster.labels_[i]:
        misclassification+=1
print('misclassification rate of Complete Linkage =', 100*misclassification/len(y),'%')


# In[160]:


# Try with another linkage
linkage=hierarchy.average(X) # create hierarchical cluster using average linkage
f,ax=plt.subplots(1,1,figsize=(20,13)) # prepare plot
plt.xlabel('Data')
plt.ylabel('Euclidean Distance')
cluster=hierarchy.dendrogram(linkage,leaf_font_size=8,ax=ax)


# In[161]:


# Supposed we want to cut in 0.5, therefore there will be 3 clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
cluster.fit_predict(X)
print("clustering result:", cluster.labels_)


# In[162]:


# In Hierarchical clustering, this library does not provide the SSE and the centroids, but we still can visualize it
# Evaluate with visualization
import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.scatter(X[:,2], X[:,3], c=cluster.labels_) # Visualize the cluster result
ax.set_title('Hierarchical Clustering  with Average Linkage')
plt.show()


# In[163]:


# Since we use supervised data, we can measure how good clustering algorithm based on its accuracy
misclassification=0
for i in range(len(y)):
    if y[i] != cluster.labels_[i]:
        misclassification+=1
print('misclassification rate of Average Linkage =', 100*misclassification/len(y),'%')


# **Density-based Clustering**
# 

# In[164]:


# Import the library and fit the model to the data. We use previous data
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.4).fit(X) # We can change eps
# See the clustering results
print('Clustering Results =', db.labels_)
# See the number of clusters in labels, ignoring noise if present. -1 is noise
n_clusters_ = len(set(db.labels_))
print('Estimated number of clusters =', n_clusters_)


# In[165]:


# In DBSCAN library, it does not provide the SSE and the centroids, but we still can visualize it
# Evaluate with visualization
import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.scatter(X[:,2], X[:,3], c=db.labels_) # Visualize the cluster result
ax.set_title('Hierarchical Clustering  with DBSCAN')
plt.show()

# Since we use supervised data, we can measure how good clustering algorithm based on its accuracy
misclassification=0
for i in range(len(y)):
    if y[i] != db.labels_[i]:
        misclassification+=1
print('misclassification rate of Average Linkage =', 100*misclassification/len(y),'%')


# In[166]:


# Number of clusters in labels, ignoring noise if present. -1 is noise
n_clusters_ = len(set(db.labels_)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)


# We see that DBSCAN actually is not really good for spherical data, but it works well in arbitrary data. Let's see the example below:

# In[169]:


from sklearn import datasets
from sklearn.preprocessing import StandardScaler
n_samples = 1500
X,y= datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
X = StandardScaler().fit_transform(X)
plt.scatter(X[:, 0], X[:, 1], s=10);


# In[170]:


# Let's evaluate it using DBSCAN VS K-means
km1 = KMeans(n_clusters=2) # run k = 2
km1.fit(X)
km1.labels_ # print the labels
f, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1],c=km1.labels_) 
ax.set_title('K-Means Clustering Results with K=2')
ax.scatter(km1.cluster_centers_[:,0], km1.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2);


# In[185]:


db = DBSCAN(eps=0.1).fit(X) # Change eps to 0.3
core_samples_mask = np.zeros_like(db.labels_, dtype=bool) # create array same size as db.labels_ with zeros
core_samples_mask[db.core_sample_indices_] = True
# See the number of clusters in labels, ignoring noise if present. -1 is noise
n_clusters_ = len(set(db.labels_))
print('Estimated number of clusters =', n_clusters_)

#Plot samples and clusters
def plot_dbscan (X,labels, core_samples_mask):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
        # White used for noise.
            col = [0, 1,1,1]
    
        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
plot_dbscan(X, db.labels_, core_samples_mask)


# # 9. ASSOCIATION RULE

# ### Preliminary
# Important Steps in AR:
# 1. Import library and dataset
# 2. Data Transformation. Dataset should be in specific format since transaction data varies
# 3. Filter Data Using Support Threshold
# 4. Create and take the rules using cofindence and lift metrics

# In[186]:


# 1
import pandas as pd # For helping in data processing
from mlxtend.preprocessing import TransactionEncoder # For data transformation
from mlxtend.frequent_patterns import apriori # For taking frequent transaction
from mlxtend.frequent_patterns import association_rules # For taking the rules
dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]


# In[187]:


# 2
te = TransactionEncoder() # Helps in encoding data using One-Hot Encoder
te_ary = te.fit(dataset).transform(dataset)
# Create new dataset in the appropirate format
df = pd.DataFrame(te_ary, columns=te.columns_)
df #view it


# In[188]:


# 3. let's assume we are only interested in itemsets that have a support of at least 60 percent.
from mlxtend.frequent_patterns import apriori
frequent_itemsets=apriori(df, min_support=0.6,use_colnames=True) #run the association rule algorithm
# frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x)) # Adding the length colum
frequent_itemsets


# In[189]:


# 4
# Take all rules
from mlxtend.frequent_patterns import association_rules
conf_rules=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
# Take only valid rules
valid_rules = conf_rules[(conf_rules['lift']>=1.01)]
valid_rules
# See rules 2 and 3, those are the same items but why different? Use analogy of motorcycle and helmet


# ### Let's see how AR helps in recommender system
# We will use Movielens dataset

# In[190]:


# 1
import pandas as pd
AR_data = pd.read_csv("dataset_AR_movies.csv")
AR_data = AR_data.iloc[:,1:]
AR_data.head() # Try to see number of data


# In[191]:


# 2
# Can you see? Different data format needs different approach for data transformation
rat=AR_data.pivot(index='userId', columns='movieId', values='rating') # Pivot is pandas's function
rat.fillna(0,inplace=True) #replace NaN with 0
rat.tail()


# In[192]:


# 2 
# Data need in the format of True/False or 1/0
# Assume we will give a good rate for the movie that has rating >= 3
rat[rat<3]=0
rat[rat>=3]=1 #Good rating
rat.tail()


# In[193]:


# 3
from mlxtend.frequent_patterns import apriori
frequent_itemsets=apriori(rat, min_support=0.3,use_colnames=True) #run the association rule algorithm
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets


# In[194]:


# 4
from mlxtend.frequent_patterns import association_rules
conf_rules=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
valid_rules = conf_rules[(conf_rules['lift']>=1.01)]
valid_rules


# What's your strategy? Assume we only take the first rule because it has highest confidence.

# In[195]:


# Check! what are the names of the movies?
print(AR_data.loc[AR_data['movieId'] == 296])
print(AR_data.loc[AR_data['movieId'] == 318])


# #### Strategy: The customer who likes to watch Pulp Fiction (1994) should be recommended to watch Shawshank Redemption (1994)!

# # 10. DEEP LEARNING
# ### Here, we will focus on how to apply deep learning for predictive analytics
# Let's start from regression by using the same dataset as used in MLR. We directly use clean data which are X_regression and y_regression.
# 
# Some important steps:
# 1. Import dataset and library
# 2. Construct the architecture of NN
# 3. Fit the model to the training data
# 4. Predict testing data
# 5. Evaluate
# 
# ### 1. DL for Regression

# In[393]:


X_regression


# In[394]:


# In Deep Learning, target data should be scale as well. This makes algorithm works better based on its mechanism.
scalerDL=MinMaxScaler()
y_regressionDL=scalerDL.fit_transform(y_regression.reshape(-1, 1))
y_regressionDL


# In[395]:


# Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regressionDL, test_size = 0.3, random_state = 0)
X_train


# In[396]:


# Import Keras Library
from tensorflow.keras.models import Sequential # This module will be used to initialize neural networks
from tensorflow.keras.layers import Dense # This module will be used to create layers in neural networks
from tensorflow.keras.optimizers import SGD # Import the optimizer method

# Initialize NN with input layer + 1 hidden layer
NN_regression = Sequential()
NN_regression.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
# units = how many nodes (neurons) are in this hidden layer.
# kernel_initializer = distribution used to initialize the weight W for each input.
# activation = we select relu which is the rectifier function.
# input_dim = how many independent variables that we have.

# Add new hidden layer
NN_regression.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# # Add new hidden layer
# NN_regression.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
 
# # Add new hidden layer
# NN_regression.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Add output layer
NN_regression.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))

# Define the optimizer and the loss function
opt = SGD(lr=0.1)
NN_regression.compile(optimizer = opt, loss = 'mean_squared_error')


# In[397]:


#Fit the model to the training data. We put the history in history variable to make learning curve
history=NN_regression.fit(X_train, y_train, batch_size = 10, epochs = 400)


# In[398]:


# Predict testing data
y_pred = NN_regression.predict(X_test)
y_pred # See that y_pred is still in normalized value, we need to inverse it


# In[399]:


y_pred=scalerDL.inverse_transform(y_pred)
y_test=scalerDL.inverse_transform(y_test)
y_pred


# In[400]:


# Then fair comparison can be done
# Evaluate Metrics
from sklearn import metrics
print("MAE =",metrics.mean_absolute_error(y_test,y_pred))
print("MSE =",metrics.mean_squared_error(y_test,y_pred))
print("R2 =",metrics.r2_score(y_test,y_pred))


# Can you compare the result with MLR result previously?

# In[401]:


# Show the learning history of DL
plt.plot(history.history['loss'])
plt.title('Learning History')
plt.ylabel('MSE')
plt.xlabel('epoch')
# plt.ylim([0, 1])
plt.show()


# In[402]:


# Show how good the prediction from visualization
plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()


# ### 2. DL for Classification
# We will use the same dataset as classification problem before which is Iris Dataset

# In[403]:


# Import dataset
df1=pd.read_csv("dataset_classification_clustering_iris.csv")
df1
# Feature Scaling using Normalization
from sklearn.preprocessing import MinMaxScaler
X=MinMaxScaler().fit_transform(df1.iloc[:,:4])
X
# Encode the target to be number
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df1["class"])
y


# In[404]:


# Because we face multi-class classification, we need to convert this into one-hot encoded
from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
# Split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
y


# In[405]:


# Import Keras Library
from tensorflow.keras.models import Sequential # This module will be used to initialize neural networks
from tensorflow.keras.layers import Dense # This module will be used to create layers in neural networks

# Initialize NN with input layer + 1 hidden layer
NN_classification = Sequential()
NN_classification.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
# units = how many nodes (neurons) are in this hidden layer.
# kernel_initializer = distribution used to initialize the weight W for each input.
# activation = we select relu which is the rectifier function.
# input_dim = how many independent variables that we have.

# Add new hidden layer
NN_classification.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# # Add new hidden layer
# NN_classification.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
 
# # Add new hidden layer
# NN_classification.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Add output layer
NN_classification.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))

# Define the optimizer and the loss function
NN_classification.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics='accuracy') # Do not forget to change the loss


# In[406]:


#Fit the model to the training data. We put the history in history variable to make learning curve
history=NN_classification.fit(X_train, y_train, batch_size = 10, epochs = 400)


# In[407]:


# evaluate the model
train_loss,train_accuracy = NN_classification.evaluate(X_train, y_train, verbose=0)
test_loss,test_accuracy = NN_classification.evaluate(X_test, y_test, verbose=0)
print("accuracy of training data:",train_accuracy)
print("accuracy of testing data:",test_accuracy)


# In[408]:


# Predict specific data
y_pred = NN_classification.predict([[0.4,0.8,0.3,0.4]])
y_pred
# Choose the class that has the highest probability


# In[411]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('Learning History')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.ylim([0, 1])
plt.show()


# **How to visualize NN Network**

# In[412]:


# !pip install nnv
from nnv import NNV

layersList = [
    {"title":"input", "units": 4, "color": "darkBlue","edges_color":"black", "edges_width":2},
    {"title":"hidden 1\n(relu)", "units": 6, "edges_width":2},
    {"title":"hidden 2\n(relu)", "units": 6, "edges_color":"red", "edges_width":2},
    {"title":"output\n(softmax)", "units": 3,"color": "darkBlue"},
]

NNV(layersList,max_num_nodes_visible=6).render()


# **Suggestion for DL Configuration:**
# 1. When we face Regression: it is suggested to use 1 Node in output layer and linear activation function. The loss function can be: mse, msle, & mae.
# 2. When we face Binary Classification: use 1 Node in output layer. If we use binary_crossentropy as loss function, then use sigmoid activation function and the target must be in 0/1. If we use hinge OR squared_hinge as loss function, then use tanh activation function and the target must be in -1/1.
# 3. When we face Multi-Class Classification: use n Node in output layer (n=number of classes). We can use categorical_crossentropy OR kullback_leibler_divergence as loss function, then use softmax activation function and the target must be in 0,1,..,n. We also can use sparse_categorical_crossentropy when we have a lot of categorical value such as in text mining so that we do not need to do one hot encoding for y data.
# 4. Practically, we can use the same activation function for all hidden layers. Some of them are: Rectified Linear Activation (ReLU), Logistic (Sigmoid), Hyperbolic Tangent (Tanh)

# # 11. OPTIMIZATION FOR DATA SCIENCE
# This chapter, we will focus on how to optimize data science technique especially for predictive analytics and supporting descriptive analytics. We will use classification dataset previously which is Iris Dataset to ease the understanding. Let's see one-by-one!

# ### FEATURE OPTIMIZATION
# In theory session, we have learned about some ways to optimize the feature. Feature engineering we have covered, Feature selection based on unsupervised approach we have covered (in EDA), Now we try to cover feature extraction and feature selection based on supervised approach.
# 
# Feature extraction will be done through application of Principal Component Analysis (PCA) and Feature selection based on supervised approach will be done through wrapper approach.
# 
# **PCA** is one of linear algebra technique that can be used to automatically perform dimensionality reduction by projecting the data to a lower dimensional subspace such that the variance of the projected data is maximized. Variance may help to retain important information from features. PCA is also the most popular technique for dimensionality reduction. It is useful when we face dataset with highly correlated. PCA can be used for numerical features, when we face categorical features, it is better to use another approach such as correspondence analysis.

# In[211]:


# Import IRIS dataset
df1=pd.read_csv("dataset_classification_clustering_iris.csv")
# Feature Scaling using Normalization
from sklearn.preprocessing import MinMaxScaler
X=MinMaxScaler().fit_transform(df1.iloc[:,:4])
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df1["class"])
print('X\n',X)
print('y\n',y)


# #### First, we will see how PCA can help supporting descriptive analytics. 
# In descriptive analytics, visualization is one of the important task. No matter what, visualization reveals what statistics cannot reveal. Whenever we face data with more than 2 or 3 dimensions, it is really hard to be catched with human eyes. Even 3D is not good enough, 2D is the best choice. In Iris data it has 4 features, but we want to visualize it only 2 features. How to do that?

# In[212]:


from sklearn.decomposition import PCA
# pca=PCA()
X_PCA=PCA(n_components=2).fit_transform(X)
X_PCA
# What do you see? All data are changed


# In[213]:


plt.scatter(X_PCA[:,0], X_PCA[:,1])
plt.show()


# From here, you can use this approach to visualize the clustering result with 2D, and you can assess either your clustering algorithm works well or not!
# 
# #### Second, let's see how PCA supports predictive analytics
# Here, I will introduce you about pipeline module in sklearn. Pipeline module incorporates some steps into single step.

# In[214]:


# evaluate pca with logistic regression algorithm
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
# define the pipeline
steps = [('pca', PCA(n_components=4)), ('m', LogisticRegression())] # TRY some n_components value!
model = Pipeline(steps=steps)
# evaluate model
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=5, n_jobs=-1)
# report performance
print('Accuracy:', n_scores.mean())


# In[215]:


# The way to automate it is by using for loop
for i in range(1,len(X[0])+1):
    # define the pipeline
    steps = [('pca', PCA(n_components=i)), ('m', LogisticRegression())] # TRY some n_components value!
    model = Pipeline(steps=steps)
    # evaluate model
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=5, n_jobs=-1)
    # report performance
    print('Accuracy of PCA',i, n_scores.mean())


# How if we want to visualize the result for better presentation?

# In[216]:


# evaluate pca with logistic regression algorithm for classification
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

# add new variables to store X (a) and Y (b) axes in plot later
a=[]
b=[]
# The way to automate it is by using for loop
for i in range(1,len(X[0])+1):
    # define the pipeline
    steps = [('pca', PCA(n_components=i)), ('m', GaussianNB())] # TRY some n_components value!
    model = Pipeline(steps=steps)
    # evaluate model
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=5, n_jobs=-1)
    # report performance
    print('Accuracy of PCA',i, n_scores.mean())
    # Store the data
    a.append(n_scores.mean())
    b.append(i)
# Plot the result    
plt.plot(b,a)
plt.xticks(b)
plt.xlabel("Number of PC's")
plt.ylabel('Accuracy')
plt.show()


# #### What do you get from 2 above examples related to number of components?
# Sometimes, PCA does not improve the performance. but sometimes it works too. Why is that? Because main advantage of PCA is that it might lose predictive information that is nonlinear since PCA only captures linear relationships between original variables.
# Therefore, the best way is to use with systematic experiment and see how it affects.
# 
# Let's say you have 4 features in dataset then you reduce it to 2. Now how to use it to predict new data? Do we still need to collect 4 features? YES, then we need to transform it based on the PCA result in training data. 

# In[217]:


steps = [('pca', PCA(n_components=2)), ('m', GaussianNB())] # TRY some n_components value!
model = Pipeline(steps=steps) # Since we use pipeline, model variable stores some models into 1
X_PCA=model[0].fit_transform(X)
model[1].fit(X_PCA,y)


# In[218]:


# make a single prediction
data = [[0.9,0.43170633,0.82646737,0.1]]
new_data=model[0].transform(data)
new_data


# In[219]:


yhat = model[1].predict(new_data)
print('Predicted Class:',yhat)


# Now we combine it into 1

# In[220]:


steps = [('pca', PCA(n_components=2)), ('m', GaussianNB())] # TRY some n_components value!
model = Pipeline(steps=steps)
model.fit(X, y)
# make a single prediction
row = [[0.9,0.43170633,0.82646737,0.1]]
yhat = model.predict(row)
print('Predicted Class:',yhat)


# ### Feature Selection
# In this part, I will introduce you to use wrapper-based method for feature selection especially using genetic algorithm (GA). For example, in IRIS dataset, we have 4 features. We can just subset some of features, e.g. we only want to try to use 1st feature and 2nd feature, then the chromosome representation will be [1,1,0,0]. 1 means we will use those features, 0 means we will not use those features. Let's see how GA is used to optimize classification algorithm based on feature selection. Do not forget to install geneticalgorithm library first: **!pip install geneticalgorithm**.

# In[221]:


# First 5 independent variables of Iris dataset
X[:5,:]


# In[222]:


# Import library
from geneticalgorithm import geneticalgorithm as ga
# Define fitness function which is ML algorithm performance: We will use Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
def fitnessfunction(decvar):
    # Constraint handling
    if np.sum(decvar)==0:
        decvar=np.array([1,1,1,1])
    # Create Logistic Regression model
    classifier = LogisticRegression()
    # Split the data using CV with K=5
    scores = cross_val_score(classifier, X[:,np.where(decvar==1)[0]], y, cv=5, scoring='accuracy') # We subset X variable
    return -1*(scores.mean()) # Evaluate the accuracy, since GA library uses minimization paradigm, we need to use minus for maxima

# Preparing GA algorithm
model=ga(function=fitnessfunction,dimension=4,variable_type='bool',algorithm_parameters = {'max_num_iteration': 100,                   'population_size':4,                   'mutation_probability':0.1,                   'elit_ratio': 0.01, # We use elitist GA as many as 1% of population
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None})
# Executing GA algorithm
model.run()


# In[223]:


# Let's test the result
decvar=model.best_variable
classifier = LogisticRegression()
scores = cross_val_score(classifier, X[:,np.where(decvar==1)[0]], y, cv=5, scoring='accuracy')
scores.mean() # Evaluate the accuracy


# **Try to analyze this result with usual logistic regression model without feature selection.**

# ### HYPER-PARAMETER TUNING
# Let's see how to optimize hyper-parameter tuning in classification problem using the same dataset previously which is iris dataset. We will cover random search and grid search in this discussion supported with scikit-learn library. By using this library, automatically model will use CrossValidation with K=5. So we do not need to split the data into training and testing.
# 
# Random Search needs number of iterations parameter since it will be used as the period of randomization and also search space can be in discrete or continuous form. However, in Grid Search, there is no number of iterations parameter because it will search all of possibilities and also search space can only be in discrete form. 
# 
# Before we start, let's discuss about what hyperparameter do we want to tune? how can we know the parameters?
# Now Let's try to implement with some algorithms, and try to compare the performance with previous result (not optimized model).

# In[224]:


# Import dataset
df1=pd.read_csv("dataset_classification_clustering_iris.csv")
# Feature Scaling using Normalization
from sklearn.preprocessing import MinMaxScaler
X=MinMaxScaler().fit_transform(df1.iloc[:,:4])
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df1["class"])


# **Random Search for SVC (Kernel, C) & KNN (weight, n_neighbors, metric)**

# In[225]:


# Random search for SVC
# Define the model
from sklearn.svm import SVC
classifier = SVC()
# define search space
space = dict()
space['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
# import distribution, not the result of randomization
from scipy.stats import uniform
space['C'] = uniform(0.1, 50)
# define search
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(classifier, space, n_iter=1000, scoring='accuracy', n_jobs=-1, random_state=1)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


# In[226]:


# Random search for SVC
# Define the model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
# define search space
space = dict()
space['weights'] = ['uniform', 'distance']
space['n_neighbors'] = range(1, 21)
space['metric'] = ['euclidean', 'manhattan', 'minkowski']
# define search
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(classifier, space, n_iter=1000, scoring='accuracy', n_jobs=-1, random_state=1)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


# **Grid Search for RF & LR**

# In[227]:


# Grid Search for Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
# define search space
space = dict()
space['n_estimators'] = [100, 300, 500, 700]
space['max_features'] = ["auto", "sqrt", "log2"]
space['criterion'] = ['entropy','gini']
# define search
from sklearn.model_selection import GridSearchCV
search = GridSearchCV(classifier, space, scoring='accuracy', n_jobs=-1)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


# In[228]:


# Grid Search for Logistic Regression Model
# Define the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression() # multi-class is auto
# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
space['C'] = [100, 10, 1.0, 0.1, 0.01]
# define search
from sklearn.model_selection import GridSearchCV
search = GridSearchCV(classifier, space, scoring='accuracy', n_jobs=-1)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


# ### PARAMETER OPTIMIZATION
# 
# Parameter Optimization is not as well-known as hyper-parameter optimization. The reason is because parameter optimization has been done automatically by ML algorithm itself. Moreover, optimizer in ML algorithm has been chosen based on its effectivity and efficiency in facing a very specific problem. But, there is no wrong to try with another optimizer. If you read the paper, sometimes it is interchangable between hyper-parameter tuning & parameter optimization because not all data scientists understand the differences between them.
# 
# In this case, we will try to optimize parameters in Multiple Linear Regression algprithm since we already tried to use MLR in previous discussion. What are the parameters of MLR in the below of formula?
# $$
# Yhat = a + b_1 X_1 + b_2 X_2 + b_3 X_3 + b_4 X_4
# $$
# 
# The idea behind parameter optimization is that we want to change local optimizer of MLR with another optimizationa algorithm such as GA. Rather than finding a, b1, b2, b3, & b4 through local optimizer, we prefer to use GA as global optimizer.

# In[423]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size = 0.3, random_state = 0)
X_train


# In[427]:


# Import library
from sklearn import metrics
from geneticalgorithm import geneticalgorithm as ga
# Define fitness function which is ML algorithm performance: We will use MLR
def fitnessfunction(decvar):
    # Create MLR model
    ylearn=decvar[0]+decvar[1]*X_train[:,0]+decvar[2]*X_train[:,1]+decvar[3]*X_train[:,2]+decvar[4]*X_train[:,3]
    # Evaluate for training dataset
    mse=metrics.mean_squared_error(y_train,ylearn)
    return (mse) # Evaluate the accuracy, GA library uses minimization paradigm, we need to use minus for maxima

# Define number of decision variables
dv=5

# Define the search space
varbound=np.array([[-30000,30000]]*dv)

# Preparing GA algorithm
model=ga(function=fitnessfunction,dimension=dv,variable_type='real',variable_boundaries=varbound,         algorithm_parameters = {'max_num_iteration': 200,                   'population_size':250,                   'mutation_probability':0.1,                   'elit_ratio': 0.01, # We use elitist GA as many as 1% of population
                   'crossover_probability': 0.9,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None})
# Executing GA algorithm
model.run()


# In[428]:


# Let's evaluate the model for testing dataset
# Create MLR model
y_pred=model.best_variable[0]+model.best_variable[1]*X_test[:,0]+model.best_variable[2]*X_test[:,1]+model.best_variable[3]*X_test[:,2]+model.best_variable[4]*X_test[:,3]
# Evaluate for training dataset
print("MAE =",metrics.mean_absolute_error(y_test,y_pred))
print("MSE =",metrics.mean_squared_error(y_test,y_pred))
print("R2 =",metrics.r2_score(y_test,y_pred))


# You can try to experiment by changing the search space, population size, number of iterations, mutation_probability, and CX_probability in order to improve the performance of GA-based MLR.

# In[429]:


plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()


# **Try to analyze this result with usual logistic regression model without feature selection.**

# ### ARCHITECTURAL DESIGN OPTIMIZATION (ADO)
# 
# ADO is very specific optimization problem that is focused on how to optimize neural network architecture. ADO is actually one of hyperparameter optimization, but this is special and a lil bit tricky because if the architecture of neural network changes, it also changes a lot of parameters inside NN such as number of weights and biases → computationally expensive!
# 
# For simplicity, here we will try to use literature suggestion on designing the architecture of NN. See the proposed formulas in PPT! The main point of this discussion is teaching you how to conduct systematic experiment. Since you can do systematic experiment, you can easily enhance the performance of algorithm based on the problem you will face later scientifically!
# 
# 1. Define the factors (variables) and levels
# - Factors: Number of hidden layers (A) and Number of hidden layer nodes (B)
# - Levels: A = 1, B = Use the first & the second formula in PPT
# 

# In[430]:


# Let's try to compare 1st formula and 2nd formula. Formulas need input:
N1=4 # Input Node
N0=3 # Output Node
# The first scenario: Use the first formula
B1=np.round(np.sqrt(N1*N0))
# The second scenario: Use the second formula
B2=np.round((N1*N0)/2)
print('B1 =',B1,'\nB2 =',B2)


# Table of Design of Experiment:
# 
# | A  |        B       |           ANN Performance |
# | ------------ | :---------------: | --------------: |
# | 1          |  `3`  |   ? |
# | 1         |  `6` |  ? |

# 2. Do experiment for each scenario -> To handle stochasticity, do multiple runs
# 3. Record the response variable and calculate the mean as well as standard deviation-> ANN Performance
# 4. Do statistical or hypothetical test to choose which one is better

# In[431]:


# Import dataset
df1=pd.read_csv("dataset_classification_clustering_iris.csv")
df1
# Feature Scaling using Normalization
from sklearn.preprocessing import MinMaxScaler
X=MinMaxScaler().fit_transform(df1.iloc[:,:4])
X
# Encode the target to be number
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df1["class"])
# Because we face multi-class classification, we need to convert this into one-hot encoded
from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
# Split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


# ### First Scenario (S1)

# In[434]:


# Import Keras Library
from tensorflow.keras.models import Sequential # This module will be used to initialize neural networks
from tensorflow.keras.layers import Dense # This module will be used to create layers in neural networks

# Multiple runs: Do 25 runs
# Create empty list to store value
S1=[]
for i in range(25):
    # Initialize NN with input layer + 1 hidden layer
    NN_classification = Sequential()
    NN_classification.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4)) #B1 here
    # Add output layer
    NN_classification.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))
    # Define the optimizer and the loss function
    NN_classification.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics='accuracy') # Do not forget to change the loss    
    # Train the ANN Model
    NN_classification.fit(X_train, y_train, batch_size = 10, epochs = 150,verbose=0) # To make it fast, just use small epoch
    # evaluate the model 
    test_loss,test_accuracy = NN_classification.evaluate(X_test, y_test, verbose=0)
    # store the accuracy of test data
    S1.append(test_accuracy)


# ### Second Scenario (S2)

# In[435]:


# Import Keras Library
from tensorflow.keras.models import Sequential # This module will be used to initialize neural networks
from tensorflow.keras.layers import Dense # This module will be used to create layers in neural networks

# Multiple runs: To make it faster, do 10 runs
# Create empty list to store value
S2=[]
for i in range(25):
    # Initialize NN with input layer + 1 hidden layer
    NN_classification = Sequential()
    NN_classification.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
    # Add output layer
    NN_classification.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))
    # Define the optimizer and the loss function
    NN_classification.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics='accuracy') # Do not forget to change the loss    
    # Train the ANN Model
    NN_classification.fit(X_train, y_train, batch_size = 10, epochs = 150,verbose=0) # To make it fast, just use small epoch
    # evaluate the model 
    test_loss,test_accuracy = NN_classification.evaluate(X_test, y_test, verbose=0)
    # store the accuracy of test data
    S2.append(test_accuracy)


# In[445]:


print("Scenario 1 Mean:",np.mean(S1))
print("Scenario 1 StDev:",np.std(S1))
print("Scenario 2 Mean:",np.mean(S2))
print("Scenario 2 StDev:",np.std(S2))
# The result might be worse than before (DL for classification) since we use smaller iteration number and smaller hidden layers


# #### Compare the result through statistical test
# Before we determine what type of statistical test we want to use, we need to check the normality of the data by using kolmogorov-smirnov. 
# - H0: Data are normally distributed
# - H1: Data are not normally distributed
# If the both results are normal (P values ≥ 0.05), Ho is accepted and we can use parametric test such as T-test to check whether these two different architectures significantly different or not. Otherwise (P values < 0.05), H0 is rejected and we can use non-parametric test such as Wilcoxon sign test test. However, some experts in statistics said that since you have data more than 25 or at least 30, your data already normally distributed.

# In[439]:


import scipy.stats as stats
# Normality test with Kolmogorov-Smirnov
print(stats.kstest(S1,'norm'))
print(stats.kstest(S2,'norm'))


# In[440]:


# Non-parametric test using Wilcoxon sign test test
stats.wilcoxon(S1,S2)


# In[444]:


# Parametric test using Paired T-test
t2, p2 = stats.ttest_rel(S1,S2)
print("t = ", t2)
print("p = ", p2)


# **INTERPRETATION**
# 
# - H0: Both scenarios are the same
# - H1: Both scenarios are significantly different
# 
# Normality test using Kolmogorov-Smirnov shows that both data are not normally distributed. Therefore, we use Wilcoxon sign test. From Wilcoxon sign test AND Paired T-Test results, we can interpret that:
# - If p-value ≥ 5%, then H0 is accepted.
# - Otherwise, H0 is rejected. 
# 
# If the result is different, we can use the better one if we focus on effectivity. But if both are the not significantly different (even tough the mean of S2 is greater than S1), we can prefer to use smaller number of hidden nodes (S1) since it gives faster computation (more efficient).
# 
# For further experiments, if we want to get a more reliable result, try to record the computational time for both scenarios and then compare again using similar test. Try to use parametric test since it has more hypothesis power. If your data is not normally distributed, try to add more runnings. If you only compare 2 samples, just use paired t-test. If you compare more than 2 samples, use ANOVA test then followed by posthoc test such as Tukey test.

# ## Congratulations, You Have Reached A Milestone!
# 
# All lessons taught in this course have given you a comprehensive data science area! From very fundamental to advanced ones! However, data science is still growing area, so many algorithms and approaches out there are being developed! What you have learned in this course is totally great choice! Data are growing and never stop to grow. Never too late to learn! 
# 
# Now, what you need to do is try to enhance your python skill, always read the data science paper/info to stay updated, and never stop to code for more than 3 months! If you want to be a full-stack data scientist, you also need to deepen database system, visualization, and communication skill. But the hardest one is already mastered! Thank you!
