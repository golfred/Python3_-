#！ Python3
#   coding:utf-8


class MathScore:
    # 假设我们给一次数学考试创建一个类，用于记录每个学生的学号、数学成绩、
    # 以及提供一个用于判断是否通过考试的check 函数
    def __init__(self, std_id, score):
        self.std_id = std_id
        self.score = score

    def check(self):
        if self.score >= 60:
            return 'pass'
        else:
            return 'failed'


xiaoming = MathScore(19, 99)

print('std_id:{}\nscore:{}\nispass:{}'.format(xiaoming.std_id, xiaoming.score, xiaoming.check()))
print('_' * 10)

xiaoming = MathScore(19, -99)

print('std_id:{}\nscore:{}\nispass:{}'.format(xiaoming.std_id, xiaoming.score, xiaoming.check()))

print('==' * 10)

'''
# 为防止出现负分数
# 将 score 变为私有，从而禁止 xiaoming.score 这样的直接调用，
# 增加一个 get_score 和 set_score 用于读写
# 这确实是种常见的解决方法，但是不得不说这简直丑爆了：
# 调用成绩再也不能使用 xiaoming.score 这样自然的方式，
# 需要使用 xiaoming.get_score() ，这看起来像口吃在说话！
# 还有那反人类的下划线和括号...那应该只出现在计算机之间窃窃私语之中...
# 赋值也无法使用 xiaoming.score = 80， 
# 而需使用 xiaoming.set_score(80)， 这对数学老师来说，太 TM 不自然了 !!!
'''


class MathScore:
    def __init__(self, std_id, score):
        self.std_id = std_id
        if score < 0:
            raise ValueError("Score can't be negative number!")
        self.__score = score

    def check(self):
        if self.__score >= 60:
            return 'pass'
        else:
            return 'failed'

    def get_score(self):
        return self.__score

    def set_score(self, value):
        if value < 0:
            raise ValueError("Score can't be negative number!")
        self.__score = value


# 作为一门简洁优雅的编程语言，Python 是不会坐视不管的，于是其给出了 Property 类
# 不管 Property 是啥，咱先看看它是如何简洁优雅的解决上面这个问题的
class MathScore:
    def __init__(self, std_id, score):
        self.std_id = std_id
        if score < 0:
            raise ValueError("Score can't be negative number!")
        self.__score = score

    def check(self):
        if self.__score >= 60:
            return 'pass'
        else:
            return 'failed'

    def __get_score__(self):
        return self.__score

    def __set_score__(self, value):
        if value < 0:
            raise ValueError("Score can't be negative number!")
        self.__score = value

    score = property(__get_score__, __set_score__)

# 与上段代码相比，主要是在最后一句实例化了一个 property 实例，
# 并取名为 score， 这个时候，我们就能如此自然的对 instance.__score 进行读写了


xiaoming = MathScore(10, 90)
print(xiaoming.score)
xiaoming.score = 80
print(xiaoming.score)
xiaoming.score = -90

'''
    它是怎么工作的呢？
    先看下 property 的参数：
    class property(fget=None, fset=None, fdel=None, doc=None)  #拷贝自 Python 官方文档
    它的工作方式：
    实例化 property 实例（我知道这是句废话）；
    调用 property 实例（比如xiaoming.score）会直接调用 fget，并由 fget 返回相应值；
    
    对 property 实例进行赋值操作（xiaoming.score = 80）则会调用 fset，
    并由 fset 定义完成相应操作；
    
    删除 property 实例（del xiaoming），则会调用 fdel 实现该实例的删除；
    
    doc 则是该 property 实例的字符说明；
    fget/fset/fdel/doc 需自定义，如果只设置了fget，则该实例为只读对象；
    这看起来和本篇开头所说的 descriptor 的功能非常相似，让我们回顾一下 descriptor：
    “descriptor 就是一类实现了__get__(), __set__(), __delete__()方法的对象。”
    
    我们知道了 property 实例的工作方式了，那么问题又来了：它是怎么实现的？
    事实上 Property 确实是基于 descriptor 而实现的，下面进入我们的正题 descriptor 吧！

'''