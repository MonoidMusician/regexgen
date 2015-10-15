# coding: utf8
import sys # for use via CLI

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

class Debug:
    EXTRA_DEBUG=None
    DEBUG=False
    #DEBUG=True
    def __init__(self):
        self._str = str()
        self._indent = "\t"
    def __iadd__(self, val):
        if self._str[-7:] == "return ":
            self._str = self._str[:-7]
        if val == 1:
            self._str += self._indent
        elif val == -1:
            self._str = self._str[:-len(self._indent)]
        else: assert val == 1 or val == -1
        return self
    def call(self, name, local=None, *sym):
        if not self.DEBUG: return
        if local is not None:
            comp = " "
            if not sym: sym = local.keys()
            for s in sym:
                comp += ("; " if comp != " " else "") + s + " = " + repr(local[s])
        else: comp = str()
        print(self._str+name+"():"+comp)
        self += 1
    def prnt(self, string):
        if not self.DEBUG: return
        print(self._str+string)
    def dump(self, local, *sym):
        if not self.DEBUG: return
        comp = ""
        if not sym: sym = local.keys()
        for s in sym:
            comp += ("; " if comp else "") + s + " = " + repr(local[s])
        print(self._str+comp)
    def ret(self, val):
        if not self.DEBUG: return val
        self += -1
        print(self._str+"return "+repr(val))
        return val
    def tailcall(self):
        self += -1
        self._str += "return "

debug = Debug()

def similar(rev, word_list, prefix):
    result = 0
    if not len(word_list[0]):
        return prefix,False
    c = word_list[0][-1 if rev else 0]
    for e in word_list:
        if c != e[-1 if rev else 0]:
            return prefix,False
    chop(rev, word_list)
    if rev:
        return c+prefix,True
    return prefix+c,True
def chop(rev, word_list):
    if rev:
        for i in range(len(word_list)):
            word_list[i] = word_list[i][:-1]
    else:
        for i in range(len(word_list)):
            word_list[i] = word_list[i][1:]
def filter_list(wordlist_left):
    if len(wordlist_left) <= 1: return [Switch(e) for e in wordlist_left]
    # Strangely enough, if this changes to a regular sort algorithm,
    # (i.e. no lambda), this fails to group "for" with "foreach" and
    # "forindex" when running with --special "keywords"
    #
    # Doing it this way pakes sure smaller words acting as whole
    # prefixies/postfixes get picked up first, and take longer words too.
    # Also sort by regular cmp(str, str) to assure a fixed order.
    wordlist_left.sort(lambda a,b: cmp(len(a), len(b)) or cmp(a,b))
    global debug
    debug.call("filter_list", locals(), "wordlist_left")
    def get_ends(string, sz):
        return string[:sz],string[-sz:]
    children = list()
    prefixes = dict()
    postfixes = dict()
    # Get a starting estimate: we work backwards from max_size-1 through 0
    # to get the longest possible prefix or postfix (the one with more
    # matches is used).
    max_size = reduce(lambda a,b: a if a >= len(b) else len(b), wordlist_left, 0)
    # This is the actual countdown
    for i in reversed(range(1,max_size)):
        #if debug.EXTRA_DEBUG: debug.call("find_matches", locals(), "i")
        # Start at the beginning of the word list; skip until long enough.
        j = 0
        while j < len(wordlist_left) and len(wordlist_left[j]) < i:
            j += 1
        # Next we start with a word and find words that start or end
        # simarly.
        while j < len(wordlist_left):
            base = wordlist_left[j]
            _pre,_post = get_ends(base, i)
            matches = [set([base]),set([base])]
            #if debug.EXTRA_DEBUG: debug.call("sub_find_matches", locals(), "j", "base", "_pre", "_post")
            for k in range(j+1,len(wordlist_left)):
                e = wordlist_left[k]
                assert len(e) >= i
                pre,post = get_ends(e, i)
                #if debug.EXTRA_DEBUG: debug.dump(locals(), "pre", "post")
                if pre == _pre: matches[0].add(e)
                if post == _post: matches[1].add(e)
            #if debug.EXTRA_DEBUG: debug.ret(matches)
            existing = [None,None]
                # If there's already a similar prefix (i.e. this is a prefix
            # of the prefix, or sub-prefix), add this to its list, as this
            # current prefixe's items will be handled later
            for pre in prefixes:
                if pre[:i] == _pre:
                    existing[0] = prefixes[pre]
                    if debug.EXTRA_DEBUG: debug.prnt("Match found (pre): "+repr(pre)+" for "+repr(_pre))
                    break
            for post in postfixes:
                if post[-i:] == _post:
                    existing[1] = postfixes[post]
                    if debug.EXTRA_DEBUG: debug.prnt("Match found (post): "+repr(post)+" for "+repr(_post))
                    break
            if len(matches[0]) + len(matches[1]) == 2:
                j += 1; continue
            # Take the one with more items
            elif len(matches[0]) >= len(matches[1]):
                match = matches[0]
                for e in match:
                    wordlist_left.remove(e)
                if existing[0]:
                    del prefixes[pre]
                    existing[0].update(match)
                    match = existing[0]
                prefixes[_pre] = match
                if debug.EXTRA_DEBUG: debug.dump(locals(), "_pre", "match")
            else:
                match = matches[1]
                for e in match:
                    wordlist_left.remove(e)
                if existing[1]:
                    del postfixes[post]
                    existing[1].update(match)
                    match = existing[1]
                postfixes[_post] = match
                if debug.EXTRA_DEBUG: debug.dump(locals(), "_post", "match")
            #if debug.EXTRA_DEBUG: debug.dump(locals(), "prefixes", "postfixes", "children")
            if not match in children: children.append(match)
        #if debug.EXTRA_DEBUG: debug.ret(children)
        if not wordlist_left: break
    else: children += [[e] for e in wordlist_left]
    # Some of these are left as sets for mutability purposes; convert them
    for i in range(len(children)):
        assert len(children[i])
        if type(children[i]) == set: children[i] = list(children[i])
        else: break
    debug.dump(locals(), "children", "prefixes", "postfixes")
    # Convert all lists of similar words to switches and potentially optimize
    children = [Switch(childL) for childL in children]
    debug.tailcall()
    return optimize(children)

class regex:
    pre = R"\("
    sep = R"\|"
    post = R"\)"
    opt = R"\?"
    pre2 = R"\["
    post2 = R"\]"
    sep2 = ""

class Switch:
    def __init__(self, word_list=None):
        # Shortcuts:
        if word_list == None: return
        if type(word_list) == str: word_list = [word_list]
        if not word_list:
            self.prefix = str()
            self.children = list()
            self.postfix = str()
            return
        elif len(word_list) == 1:
            self.prefix = word_list[0]
            self.children = list()
            self.postfix = str()
            return
        # Else we take the long way:
        global debug
        debug.call("Switch.__init__", locals(), "word_list")
        # Sort: shortest first with fixed order too
        word_list.sort(lambda a,b: cmp(len(a), len(b)) or cmp(a,b))
        debug.dump(locals(), "word_list")
        smallest = word_list[0]
        middle = list(word_list)
        
        # Find a common prefix and common postfix
        prefix = str()
        while len(prefix) < len(smallest):
            (prefix,s) = similar(False, middle, prefix)
            if not s: break
        
        postfix = str()
        while len(prefix)+len(postfix) < len(smallest):
            (postfix,s) = similar(True, middle, postfix)
            if not s: break
        debug.dump(locals(), "middle")
        # And return to filter_list to sort through what's left
        if len(prefix) or len(postfix):
            self.children = filter_list(middle)
        else: self.children = [Switch(m) for m in middle]
        self.prefix = prefix
        self.postfix = postfix
        debug.ret(self)
    def joinwpre_post(self, pre, post):
        j = self.join()
        skip = 0
        slash = False
        if self.issimple(): simple = True
        elif j.find(regex.sep) == None: simple = True
        else:
            for i in range(len(j)):
                if slash:
                    if j[i] == "(": skip += 1
                    elif j[i] == ")": skip -= 1
                    elif not skip and j[i] == "|":
                        simple = False
                        break
                    slash = False
                elif j[i] == "\\": slash = not slash
            else: simple = True
        if simple:
            return pre+j+post
        else: return pre+regex.pre+j+regex.post+post
    def join(self):
        #debug.call("Switch.join", locals(), "self")
        children = self.children
        # For simple lists like c(ompile|aller) or (ha|po)t,
        # add the prefix or postfix to each one instead of
        # using grouping
        if (len(children) > 1
           and len(self.prefix)+len(self.postfix) == 1
           and not self.hassingle()
           and not self.hasopt()):
            children = [c.joinwpre_post(self.prefix,self.postfix) for c in children]
            #debug.dump(locals(), "children")
            #return debug.ret(sep.join(children))
            return regex.sep.join(children)
        middle = None
        if len(children) == 0:
            middle = str()
        elif len(children) == 1:
            middle = children[0].join()
        elif self.hasopt():
            if len(children) == 2:
                otherchild = self.exceptforopt()[0]
                if otherchild.issingle():
                    middle = otherchild.prefix+regex.opt
                else:
                    middle = regex.pre+otherchild.join()+regex.post+regex.opt
            else:
                if self.isallsingle(allow_opt=1):
                    middle = regex.pre2+regex.sep2.join(e.prefix for e in children)+regex.post2+regex.opt
                elif len(children) < 5:
                    middle = regex.pre+regex.sep.join(e.join() for e in children if not e.isopt())+regex.post+regex.opt
            #else: fall_through
        elif self.isallsingle():
            middle = regex.pre2+regex.sep2.join(e.prefix for e in children)+regex.post2
        if middle == None:
            middle = regex.pre+regex.sep.join(e.join() for e in children)+regex.post
        #return debug.ret(prefix+middle+postfix)
        return self.prefix+middle+self.postfix
    def __repr__(self):
        if self.isopt(): return '/opt'
        elif self.issimple(): return self.prefix.__repr__()
        #elif self.children: return [self.prefix, self.children, self.postfix].__repr__()
        #else: return "["+repr(self.prefix)+", /null, "+repr(self.postfix)+"]"
        else: return repr(self.prefix)+"+{"+", ".join([repr(e) for e in self.children])+"}+"+repr(self.postfix)
    def __eq__(self, other):
        return self.__dict__.__eq__(other.__dict__)
    def isopt(self):
        return (len(self.prefix) == 0
                and not len(self.children)
                and not len(self.postfix))
    def issingle(self):
        return (len(self.prefix) == 1
                and not len(self.children)
                and not len(self.postfix))
    def issimple(self):
        return (not len(self.children)
                and not len(self.postfix))
    def hasopt(self):
        for i in self.children:
            if i.isopt(): return True
        return False
    def hassingle(self):
        for i in self.children:
            if i.issingle(): return True
        return False
    def hassimple(self):
        for i in self.children:
            if i.issimple(): return True
        return False
    def isallsingle(self, allow_opt=False):
        for i in self.children:
            if allow_opt and i.isopt(): continue
            if not i.issingle(): return False
        return True
    def exceptforopt(self):
        return [e for e in self.children if not e.isopt()]

def optimize(children):
    global debug
    debug.call("optimize", locals(), "children")
    globs = ListToListset(children)
    children2 = list()
    optimized = list()
    for map_instance in globs:
        map_instances = map_instance.split()
        children2 += map_instances[0].connections
        for e in map_instances[1:]:
            assert len(e) > 1
            element = Switch()
            element.children = e.children
            element.prefix  = Switch(sorted(e.prefixes.keys())).join()
            element.postfix = Switch(sorted(e.postfixes.keys())).join()
            debug.dump(locals(), "element")
            optimized.append(element)
    debug.dump(locals(), "optimized")
    if optimized: children2.extend(optimize(optimized))
    for i in children2:
        if i.isopt():
            children2.remove(i)
            children2 = [i]+children2 # Switch.join() assumes this is at the beginning
            break
    return debug.ret(children2)

# FIXME: merge the next two classes; really should be a triangle
# of potential children,prefix,postfix connections.
class ListToListset:
    def __init__(self, init):
        self.childrenL = list()
        self.elementsL = list()
        for e in init: self.add(e)
    def add(self, child):
        try:
            if not child.children: raise ValueError()
            self.find(child.children).add(child)
        except ValueError:
            self.childrenL.append(child.children)
            self.elementsL.append(PrePostMap(child))
    def find(self, childL):
        i = self.childrenL.index(childL)
        return self.elementsL[i]
    def __iter__(self): return iter(self.elementsL)
    def __len__(self): return len(self.elementsL)

class PrePostMap:
    def __init__(self, element=None):
        self.prefixes = dict()
        self.postfixes = dict()
        self.connections = list()
        self.children = element.children if element else None
        if element != None: self.add(element)
    def add(self, element):
        assert not len(element.children) or element.children == self.children
        if element.prefix in self.prefixes:
            self.prefixes[element.prefix] += 1
        else:
            self.prefixes[element.prefix] = 1
        if element.postfix in self.postfixes:
            self.postfixes[element.postfix] += 1
        else:
            self.postfixes[element.postfix] = 1
        self.connections.append(element)
    def references(self, pre, string):
        return (self.prefixes[string]
           if pre else self.postfixes[string])
    def remove(self, pre, string):
        if pre: del self.prefixes[string]
        else: del self.postfixes[string]
        i = 0
        while i<len(self.connections):
            e = self.connections[i]
            if pre: effix = e.prefix
            else: effix = e.postfix
            if effix == string:
                self.connections.remove(e)
                if pre:
                    self.postfixes[e.postfix] -= 1
                    if not self.postfixes[e.postfix]:
                        del self.postfixes[e.postfix]
                else:
                    self.prefixes[e.prefix] -= 1
                    if not self.prefixes[e.prefix]:
                        del self.prefixes[e.prefix]
            else: i += 1
    def removeconnection(self, child):
        for e in self.connections:
            if e == child:
                self.connections.remove(e)
                self.postfixes[e.postfix] -= 1
                if not self.postfixes[e.postfix]:
                    del self.postfixes[e.postfix]
                self.prefixes[e.prefix] -= 1
                if not self.prefixes[e.prefix]:
                    del self.prefixes[e.prefix]
                return
    # Grow algorithm
    def split(self):
        result = [PrePostMap()] # discarded "other/rest" elements
        result[0].prefixes  = dict(self.prefixes)
        result[0].postfixes = dict(self.postfixes)
        result[0].connections = list(self.connections)
        result[0].children = self.children
        disc = result[0]
        if not len(self.children): return result
        global debug
        debug.call("PrePostMap.split", locals(), "self")
        assert disc.connections[0] in disc.connections
        assert disc.hasconnection(disc.connections[0].prefix, disc.connections[0].postfix)
        _i = 0
        while _i < len(disc.prefixes):
            debug.dump(locals(), "disc")
            pre = disc.geteffix(True, _i)
            prae = [pre]; posti = []
            for post in disc.geteffix(False):
                if disc.hasconnection(pre,post):
                    posti.append(post)
                    break
            assert posti
            for pre,effix in disc.geteffix():
                if pre and effix in prae: continue
                elif not pre and effix in posti: continue
                if pre:
                    temp = [prae+[effix],posti]
                else:
                    temp = [prae,posti+[effix]]
                if disc.isintegral(*temp):
                    if pre:
                        prae.append(effix)
                    else:
                        posti.append(effix)
            if len(prae) > 1 or len(posti) > 1:
                result.append(PrePostMap())
                result[-1].children = disc.children;
                connections = list()
                for pre in prae:
                    for post in posti:
                        connections.append(disc.makeconnection(pre,post))
                        disc.removeconnection(connections[-1])
                        result[-1].add(connections[-1])
                debug.dump(locals(), "disc", "result")
            else: _i += 1
        return debug.ret(result)
    def __len__(self): return len(self.connections)
    def geteffix(self, pre=None, i=None):
        if pre is None:
            prefixi  = list(self.prefixes.keys())
            prefixi  = zip([True]*len(prefixi),prefixi)
            postfixi = list(self.postfixes.keys())
            postfixi = zip([False]*len(postfixi),postfixi)
            effixi = prefixi + postfixi
            effixi.sort(lambda a,b:
              cmp(self.references(*a), self.references(*b)))
            if i is not None:
                return effixi[i][1]
            else: return effixi # warning!: each entry of the form [pre?, string]
        else:
            effixi = list(self.prefixes.keys() if pre else self.postfixes.keys())
            effixi.sort(lambda a,b:
              cmp(self.references(pre, a), self.references(pre, b)))
            if i is not None:
                return effixi[i]
            else: return effixi
    def isfinal(self):
        test = Switch()
        test.children = self.children
        for i in self.prefixes:
            test.prefix = i
            for j in self.postfixes:
                test.postfix = j
                if not test in self.connections:
                    return False
        return True
    def hasconnection(self, *arg):
        test = self.makeconnection(*arg)
        for c in self.connections:
            assert c.children == self.children
            if c == test: return True
        return False
    def makeconnection(self, pre, post):
        test = Switch()
        test.children = self.children
        test.prefix = pre
        test.postfix = post
        return test
    def isintegral(self, prae, posti):
        for i in prae:
            for j in posti:
                if not self.hasconnection(i,j): return False
        return True
    def __repr__(self): return repr(self.__dict__)

def swap_escapes(char, string):
    spilt = string.split("\\"+char)
    for i in range(len(spilt)):
        spilt[i] = spilt[i].replace(char, "\\"+char)
    return char.join(spilt)

def main():
    global debug
    PRINT_RESULT = False
    word_list = sys.argv[1:]
    if word_list:
        if word_list[0] == "--EXTRA_DEBUG" or word_list[0] == "-e":
            word_list = word_list[1:]
            debug.DEBUG = True
            debug.EXTRA_DEBUG = True
        elif word_list[0] == "--DEBUG" or word_list[0] == "-d":
            word_list = word_list[1:]
            debug.DEBUG = True
        elif word_list[0] == "--RESULT" or word_list[0] == "-r":
            word_list = word_list[1:]
            PRINT_RESULT = True
    if len(word_list) == 2 and word_list[0] == "--special":
        word_list = {
            "builtin_functions": builtin_functions,
            "keywords": keywords,
            "operators": operators
        }[word_list[1]]
    elif word_list == ["-"]:
        word_list = []
        while True:
            l = sys.stdin.readline()
            if not l: break
            l = l.rstrip() # strip trailing spaces and \n
            if l: word_list.append(l) # ignore blank lines
    word_list = list(set(word_list)) # protect against duplicates
    result = filter_list(word_list)
    if PRINT_RESULT: print(repr(result))
    result = regex.sep.join(sw.join() for sw in result)
    for char in (".","(","|",")","?","[","]","+","*","{","}"):
        result = swap_escapes(char, result)
    result = result.replace("&", "&amp;")
    result = result.replace("<", "&lt;")
    result = result.replace(">", "&gt;")
    return result

if __name__ == "__main__":
    print(main())
