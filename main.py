import numpy as np

print("PART 1. FOR LOOPS")
#1
my_name = "Nguyễn Xuân Lộc"
for c in my_name:
    print(c)

print('\n\n\n')
#2
for i in range(1, 11):
    print(i)
#3a
sum = 0
for i in range(1, 11):
    sum += i

print('a) Tổng là: ', sum)
#3b
sum = 0
for i in range(1, 6):
    sum += i

print('b) Tổng là: ', sum)
#4
mydict = {"a": 1, "b": 2, "c": 3, "d": 4}
#a
for e in mydict:
    print(e)
#b
for e in mydict:
    print(mydict[e])
#c
for e in mydict:
    print(e, ":", mydict[e])

#5


def merge_list_to_tuple(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


courses = [131, 141, 142, 212]
names = ["Maths", "Physics", "Chem", "Bio"]
for i in merge_list_to_tuple(courses, names):
    print(i)

#6
#a
str = "jabbawocky"
vowels = "ueoai"
number_of_consonant = 0

for c in str:
    if c not in vowels:
        number_of_consonant += 1

print("a) number of consonant in given string:", number_of_consonant)
#b
number_of_consonant = 0

for c in str:
    if c in vowels:
        continue
    number_of_consonant += 1

print("b) number of consonant in given string:", number_of_consonant)

try:
    for a in range(-2, 2):
        print("10/a =", 10.0/a)
except:
    print("can't divided by zero")

#8
ages = [23, 10, 80]
names = ["Hoa", "Lam", "Nam"]

tuple_age_name = merge_list_to_tuple(ages, names)

tuple_age_name.sort(key=lambda a: a[0])
print(tuple_age_name)


#9
#a
firstnames = open('firstname.txt', 'r')
#b
for line in firstnames:
    print(line)

#c
print("c)")
firstnames = open('firstname.txt', 'r')
print(firstnames.read())


print("PART 2. DEFINE A FUNCTION")
#1
def sum(*kwargs):
    a = 0
    for i in kwargs:
        a += i
    return a

print("1) sum =", sum(3, 4, 5))
#2
# Create the matrix M
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Create the vector v
v = np.array([1, 2, 3])

# Check the rank and shape of the matrix M
rank_M = np.linalg.matrix_rank(M)
shape_M = M.shape

# Check the rank and shape of the vector v
rank_v = np.linalg.matrix_rank(v)
shape_v = v.shape

print("2) Matrix M:")
print(M)
print("Rank of M:", rank_M)
print("Shape of M:", shape_M)

print("\nVector v:")
print(v)
print("Rank of v:", rank_v)
print("Shape of v:", shape_v)

#3
new_matrix = np.add(M, 3)
print("3) New matrix:")
print(new_matrix)

#4
M_transpose = M.T
print("4) M transpose:")
print(M_transpose)

v_transpose = v.reshape(-1, 1)
print("v transpose:")
print(v_transpose)

#5
x = np.array([2, 7])
x_norm = np.linalg.norm(x)
x_normalization = x/x_norm

print("5) norm of vector x:", x_norm)
print("Normalization of vector x", x_normalization)

#6
a = np.array([10, 15])
b = np.array([8, 2])
c = np.array([1, 2, 3])

try:
    print("6) a + b =", np.add(a, b))
except:
    print("Error!")

try:
    print("a - b =", np.subtract(a, b))
except:
    print("Error!")

try:
    print("a - c =", np.subtract(a, c))
except:
    print("Error! Because it is not same dimension size")


#7
print("7) Dot product of a and b:")
print(np.dot(a, b))

#8
print("8)")
A = np.array([[2, 4, 9], [3, 6, 7]])

# a/ Rank of matrix A
rank_A = np.linalg.matrix_rank(A)
print("Rank of A:", rank_A)

# Shape of matrix A
shape_A = A.shape
print("Shape of A:", shape_A)

# b/ Get the value 7 in A
value_7 = A[1, 2]
print("Get value 7 in A:", value_7)

# c/ Return the second column of A
second_column = A[:, 1]
print("Second column of A:", second_column)

#9
random_matrix = np.random.randint(-10, 10, size=(3, 3))
print("9) Random 3x3 matrix:")
print(random_matrix)

#10
identity_matrix = np.eye(3)
print("10) Identity matrix:")
print(identity_matrix)
#11
random_matrix = np.random.randint(1, 10, size=(3, 3))
print("11) random matrix:\n", random_matrix)
print("a) Trace of matrix using one command:", np.trace(random_matrix))


def trace(matrix):
    trace = 0
    for i in range(len(matrix[0])):
        trace += matrix[i, i]
    return trace


print("b) Trace of matrix using for loop:", trace(random_matrix))

#12
random_matrix = np.random.randint(1, 10, size=(3, 3))
for i in range(3):
    random_matrix[i, i] = i + 1
print("12) Matrix with main diagonal 1, 2, 3:")
print(random_matrix)

#13
A = np.array([[1, 1, 2], [2, 4, -3], [3, 6, -5]])
print("13) Determinant of matrix A:", np.linalg.det(A))

#14
a1 = [1, -2, -5]
a2 = [2, 5, 6]
M = np.array([a1, a2])
M = M.T
print("14)")
print(M)

#15
import matplotlib.pyplot as plt

# Generate values for y in the range (-5 <= y < 6)
y_values = range(-5, 6)

# Compute the square of each y value
y_squared = [y ** 2 for y in y_values]

# Plot the square of y
plt.plot(y_values, y_squared)
plt.title("Square of y")
plt.xlabel("y")
plt.ylabel("y^2")
plt.grid(True)
plt.show()

#16
four_evenly_spaced = [i for i in range(0, 33, 4)]
print("16) c1.", four_evenly_spaced)
print("c2.", np.linspace(0, 32, int(32/4) + 1))
#17
x = np.linspace(-5, 5, 50)
y = [i**2 for i in x]
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#18
x = np.linspace(-5, 5, 50)
y = [np.exp(i) for i in x]
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("exp(x)")
plt.title("y = exp(x)")
plt.show()

#19
x = np.linspace(0, 5, 50)
y = [np.log(i) for i in x]
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("log(x)")
plt.title("y = log(x)")
plt.show()

#20
#plot 1:
y = [np.exp(i) for i in x]
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("exp(x)")
plt.title("y = exp(x)")

#plot 2:
y = [np.exp(i * 2) for i in x]
plt.subplot(1, 2, 2)
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("exp(2*x)")
plt.title("y = exp(2*x)")

plt.show()
#20 2
#plot 1:
y = [np.log(i) for i in x]
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("log(x)")
plt.title("y = log(x)")

#plot 2:
y = [np.log(i * 2) for i in x]
plt.subplot(1, 2, 2)
plt.plot(x,y)

plt.xlabel("x")
plt.ylabel("log(2*x)")
plt.title("y = log(2*x)")
plt.show()