import pandas as pd
import matplotlib.pyplot as plt
import timeit
from functools import lru_cache
import pickle
import os

#   2.b) Usuniecie duplikatów
df = pd.read_csv('train.csv')
df = df.drop_duplicates()
df.to_csv('no_duplicates.csv', index = False)

#   2.c) Korelacja miedzy limitem kredytu a wiekiem
korelacja = df['limit_bal'].corr(df['age'])
print(f"Korelacja między limit_bal a age: {korelacja}\n")

#   2.d)Dodanie nowej kolumny będącą sumą wszystkich transakcji
bill_columns = [col for col in df.columns if 'bill_amt' in col]
df['total_bill_amt'] = df[bill_columns].sum(axis=1)
df.to_csv('new_column.csv', index = False)

#   2.e) 10 najstarszych klientów i narysuj tabelkę w której będą znajdować się tylko kolumny:
#   limit_bal, age, education (po nazwie), oraz nowo dodana kolumna
oldest_clients = df.nlargest(10, 'age')[['limit_bal', 'age', 'total_bill_amt']]
print(f'{oldest_clients}\n')

#   2.f) Narysuj w jednym oknie (subplots) histogram limitu kredytu, wieku, oraz zależność limitu kredytu od wieku
fig, axes = plt.subplots(3, 1, figsize=(15,30))

df['limit_bal'].plot(kind='hist', ax=axes[0], title='Histogram Limitu Kredytu')
df['age'].plot(kind='hist', ax=axes[1], title='Histogram Wieku')
axes[2].scatter(df['age'], df['limit_bal'])
axes[2].set_title('Zależność Limitu Kredytu od Wieku')
axes[2].set_xlabel('Wiek')
axes[2].set_ylabel('Limit Kredytu')

plt.ticklabel_format(style='plain')
plt.show()

#   3.a) Wzbogać klasę Tree o dekorator @property do odczytywania najmniejszej wartości w całym drzewie
class Node:
    def __init__(self, value=None):
        self.value = value
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.value) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

class Tree:
    def __init__(self, root=None):
        self.root = root

    def goThrough(self, node = None):
        nodes = []
        if node == None:
            if self.root:
                nodes.append(self.root.value)
                nodes.extend(self.goThrough(self.root))
            return nodes
        else:
            for child in node.children:
                nodes.append(child.value)
                nodes.extend(self.goThrough(child))
            return nodes

    @property
    def min_value(self):
        all_values = self.goThrough()
        if all_values:
            return min(all_values)
        else:
            return None

    def __str__(self):
        if self.root:
            return str(self.root)
        else:
            return "Empty Tree"

#Tworzenie korzenia
root = Node('Root')
tree = Tree(root)

#Tworzenie drzewa, dodawnie węzłów
childA = Node('Child A')
childB = Node('Child B')
root.add_child(childA)
root.add_child(childB)

subchildC = Node('Subchild C')
subchildD = Node('Subchild D')
childA.add_child(subchildC)
childA.add_child(subchildD)

subchildE = Node('Subchild A')
subchildF = Node('Subchild F')
childB.add_child(subchildE)
childB.add_child(subchildF)

subsubchildG = Node('SubSubchild G')
subsubchildH = Node('SubSubchild H')
subsubchildI = Node('SubSubchild I')

subchildD.add_child(subsubchildG)
subchildD.add_child(subsubchildH)
subchildD.add_child(subsubchildI)

print(tree)

min_value = tree.min_value
print(f"Najmniejsza wartość w drzewie: {min_value}\n")

#   3.b) Zaimplementuj funkcję do obliczania kolejnych elementów ciągu Fibonacciego w sposób rekurencyjny,
#   zmierz jej czas działania używając biblioteki timeit, następnie użyj dekoratora @lru_cache,
#   i zmierz czas ponownie

def fib_recursive(n):
    if n <= 1:
        return n
    else:
        return fib_recursive(n-1) + fib_recursive(n-2)

time_recursive = timeit.timeit(lambda: fib_recursive(30), number=1)
print(f"Czas obliczeń rekurencyjnych: {time_recursive} s")

@lru_cache(maxsize=None)
def fib_with_cache(n):
    if n <= 1:
        return n
    else:
        return fib_with_cache(n-1) + fib_with_cache(n-2)

time_cached = timeit.timeit(lambda: fib_with_cache(30), number=1)
print(f"Czas obliczeń z użyciem @lru_cache: {time_cached} s\n")

#   3.c) napisz własny dekorator który zapisze na dysku wynik działania funkcji
#   i przy kolejnym użyciu wczyta go z dysku zamiast obliczać ponownie
def cache_to_disk(func):
    def wrapper(*args, **kwargs):
        filename = f"{func.__name__}_cache.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                result = pickle.load(file)
            print(f'Wczytane pliku {func.__name__}_cache.pkl')
        else:
            result = func(*args, **kwargs)
            with open(filename, 'wb') as file:
                pickle.dump(result, file)
            print(f'Tworzenie pliku {func.__name__}_cache.pkl')
        return result
    return wrapper

#Przykładowa funkcja do zastosowania dekoratora
@cache_to_disk
def example_function():
    return [i**2 for i in range(10)]

#Test dekoratora
time_without_file = timeit.timeit(lambda: example_function(), number=1)
time_with_file = timeit.timeit(lambda: example_function(), number=1)

#result = example_function()
#print(result)

os.remove('example_function_cache.pkl')

