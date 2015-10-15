# coding: utf8

builtin_functions = [
  # Standard
  'append', 'setsize', 'subvec', 'substr', 'contains', 'int', 'delete',
  'int', 'num', 'keys', 'pop', 'size', 'streq', 'sort', 'sprintf',
  'printf', 'print', 'find', 'split', 'rand', 'id', 'typeof', 'die',
  'compile', 'call', 'caller', 'closure', 'bind',
  # Regex
  'regex.comp', 'regex.exec',
  # Math
  'math.sin', 'math.cos', 'math.exp', 'math.ln', 'math.sqrt',
  'math.atan2', 'math.e', 'math.pi',
  # Bits
  'bits.sfld', 'bits.fld', 'bits.setfld', 'bits.buf',
  # Unicode
  'utf8.chstr', 'utf8.strc', 'utf8.substr', 'utf8.size', 'utf8.validate',
  # I/O
  'io.open', 'io.close', 'io.read', 'io.write', 'io.seek', 'io.tell',
  'io.readln', 'io.stat', 'io.readfile', 'io.load_nasal',
  # Threading
  'thread.newthread', 'thread.newlock', 'thread.lock', 'thread.unlock',
  'thread.newsem', 'thread.semdown', 'thread.semup',
  # Unix/commandline interaction
  'unix.pipe', 'unix.fork', 'unix.dup2', 'unix.exec', 'unix.waitpid',
  'unix.opendir', 'unix.readdir', 'unix.closedir', 'unix.time', 'unix.chdir',
  'unix.environ',
  # SQLite databases
  'sqlite.open', 'sqlite.close', 'sqlite.prepare', 'sqlite.exec', 'sqlite.finalize',
  # FlightGear/SimGear
  'setlistener', 'cmdarg', 'removelistener', 'fgcommand', 'addcommand',
  'removecommand', 'logprint', 'printlog', 'setprop', 'getprop',
  'interpolate', 'settimer', 'maketimer', 'isa', 'defined'
]

keywords = [
  'if', 'elsif', 'else', 'for', 'while', 'foreach', 'forindex',
]
keyword_regex = R'break(\s+[A-Z]{2,16})?|continue(\s+[A-Z]{2,16})?|return|([A-Z]{2,16})(?=\s*;([^\)#;]*?;){0,2}[^\)#;]*?\))'

reserved_id = [
  'me', 'arg', 'parents'
]

operators = [
  '!', '*', '-', '+', '~', '/', '==', '=', '!=', '<=',
  '>=', '<', '>', '?', ':', '*=', '/=', '+=',
  '-=', '~=', '...'
]
operator_words = [ 'and', 'or' ]

pre = R"\("
sep = R"\|"
post = R"\)"
opt = R"\?"

class Indent:
    def __init__(self):
        self._str = str()
        self._indent = "  "
    def __iadd__(self, val):
        if val == 1:
            self._str += self._indent
        elif val == -1:
            self._str = self._str[:-len(self._indent)]
        else: assert val == 1 or val == -1
        return self
    def call(self, name, local, *sym):
        comp = str()
        for s in sym:
            comp += s + " = " + repr(local[s])
        print(self._str+name+"(): "+comp)
    def dump(self, local, *sym):
        comp = str()
        for s in sym:
            comp += s + " = " + repr(local[s])
        print(self._str+comp)
    def ret(self, val):
        print(self._str+"return "+repr(val))

indent = Indent()

def similar(rev, word_list, prefix):
    result = 0
    if not len(word_list[0]):
        return prefix,False
    c = word_list[0][-1 if rev else 0]
    for e in word_list:
        if c != e[-1 if rev else 0]:
            return prefix,False
    chop(rev, word_list)
    return prefix+c,True
def chop(rev, word_list):
    if rev:
        for i in range(len(word_list)):
            word_list[i] = word_list[i][:-1]
    else:
        for i in range(len(word_list)):
            word_list[i] = word_list[i][1:]
def filter_list(wordlist_left):
    if not len(wordlist_left): return list()
    children1 = group(False, wordlist_left)
    children2 = group(True, wordlist_left)
    if len(children1) >= len(children2):
        children = children1
    else: children = children2
    for i in range(len(children)):
        children[i] = Switch(children[i])
    return children
def group(rev, wordlist_left):
    size = len(wordlist_left)
    children = list()
    i = 0
    while i < size:
        if i == 0 and not len(wordlist_left[i]):
            c = None
        else: c = wordlist_left[i][-1 if rev else 0]
        for j in range(i+1,size+1):
            if j == size: break
            elif wordlist_left[j][-1 if rev else 0] != c: break
        children.append(wordlist_left[i:j])
        i = j
    return children

class Switch:
    def __init__(self, word_list):
        # Shortcuts:
        if not len(word_list):
            self.children = list()
            self.prefix = str()
            self.postfix = str()
            return
        elif len(word_list) == 1:
            self.children = list()
            self.prefix = word_list[0]
            self.postfix = str()
            return
        # Else we take the long way:
        global indent
        indent += 1
        indent.call("Switch.__init", locals(), "word_list")
        smallest = reduce(lambda a,b: a if len(a) < len(b) else b, word_list)
        middle = list(word_list)
        
        prefix = str()
        while len(prefix) < len(smallest):
            indent.dump(locals(), "middle")
            (prefix,s) = similar(False, middle, prefix)
            if not s: break
            indent.dump(locals(), "middle")
        
        postfix = str()
        while len(prefix)+len(postfix) < len(smallest):
            indent.dump(locals(), "middle")
            (postfix,s) = similar(True, middle, postfix)
            if not s: break
            indent.dump(locals(), "middle")
        
        self.children = filter_list(middle)
        self.prefix = prefix
        self.postfix = postfix
        indent += -1
    def join(self):
        children = self.children
        middle = None
        if len(children) == 0:
            middle = str()
        elif len(children) == 1:
            middle = children[0].join()
        elif children[0].isopt():
            print(children[1].issimple().__repr__())
            if len(children) == 2 and children[1].issimple():
                middle = children[1].prefix+opt
            #else: fall_through
        if middle == None:
            joined = children
            for i in range(len(children)):
                joined[i] = children[i].join()
            middle = pre+sep.join(joined)+post
        return self.prefix+middle+self.postfix
    def issimple(self):
        return (len(self.prefix) == 1
                and not len(self.children)
                and not len(self.postfix))
    def isopt(self):
        return (len(self.prefix) == 0
                and not len(self.children)
                and not len(self.postfix))

def swap_escapes(char, string):
    spilt = string.split("\\"+char)
    for i in range(len(spilt)):
        spilt[i] = spilt[i].replace(char, "\\"+char)
    return char.join(spilt)

def generate():
    word_list=[
        "ioi", "ii", "iik"
    ]
    word_list.sort()
    result = Switch(word_list).join()
    for char in (".","(","|",")","?","[","]","+","*","{","}"):
        result = swap_escapes(char, result)
    result = result.replace("&", "&amp;")
    result = result.replace("<", "&lt;")
    result = result.replace(">", "&gt;")
    return result

def main():
    return generate()

if __name__ == "__main__":
    print(main())
