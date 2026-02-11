def non_dup(array):
    return list(set(array))
    # for num in array:
    #     num_set.add(num)
        
    # return list(num_set)

print(non_dup([1, 1, 1, 3, 5, 6]))
print(non_dup([1, 3, 1, 3, 2, 6]))

class Dictionary():
    def __init__(self):
        self.array = []
        for i in range(1000):
            self.array.append([])
        self.new_value = flat()
        self.version
        
    def set(self, key, value):
        index = (hash(key)) % (len(self.array))
        self.array[index].append([key, value])
        self.nonce += 1
        
    def get(self, key):
        index = (hash(key)) % (len(self.array))
        
        for pairs in self.array[index]:
            if pairs[0] == key:
                return pairs[1]
        
        return None
    
    def update_all(self, value):

arr = ["", [10, 0] [20, 0], ""]

arr.set('a', 10)
arr.set('b', 20)
arr.set_all(30)



