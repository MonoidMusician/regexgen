# coding: utf8
import sys # for use via CLI
import fileinput # simple inputting of files and stdin
                 # (used when we don't have a wordlist
                 # and for each --file=/-f argument).

OPTIMIZE=False

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

# Because we are often comparing lists, i.e. positional, make sure
# we have a fixed ordering. Additionally, this puts smallest first,
# as expected by both filter_list and Switch.__init__
def sort_pred(a,b): return cmp(len(a), len(b)) or cmp(a,b)

# Helper class for handling algorithm debugging.
#
# Run this script with -d/--DEBUG or -e/--EXTRA_DEBUG
# to view the output
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
    # Indicate a function call; requires the name of the function
    # (e.g. "Switch.join") and can optionally show parameters; ex:
    #   def fn(*args, **kwargs):
    #       debug.call("fn", locals()) # prints both args and kwargs
    def call(self, name, local=None, *sym):
        if not self.DEBUG: return
        if local is not None:
            comp = " "
            if not sym: sym = local.keys()
            for s in sym:
                assert type(s) == str
                comp += ("; " if comp != " " else "") + s + " = " + repr(local[s])
        else: comp = str()
        print(self._str+name+"():"+comp)
        self += 1
    # Print a message with the current indent
    def prnt(self, string):
        if not self.DEBUG: return
        print(self._str+str(string))
    # Dump local symbols
    def dump(self, local, *sym):
        if not self.DEBUG: return
        comp = ""
        if not sym: sym = local.keys()
        for s in sym:
            assert type(s) == str
            comp += ("; " if comp else "") + s + " = " + repr(local[s])
        print(self._str+comp)
    # Print a return value; passes value back for syntax like:
    #   return debug.ret(...)
    def ret(self, val):
        if not self.DEBUG: return val
        self += -1
        print(self._str+"return "+repr(val))
        return val
    # Set up the next call as a tail call, e.g.:
    #   debug.tailcall()
    #   return fn(...)
    def tailcall(self):
        self += -1
        self._str += "return "
    # In case of a degenerate case or such after a tail call,
    # this makes sure that the output indicates what was returned
    # without full output of arguments and values.
    #   val = local[sym] if sym is not None else val = local
    # Returns the "val"; doesn't print anything if not following
    # a declared tail call.
    def cancelcall(self, name, local, sym=None):
        tailcall = (self.DEBUG and self._str[-7:] == "return ")
        if sym is None:
            val = local
            if tailcall:
                print(self._str+name+"(): return "+repr(val))
        else:
            val = local[sym]
            if tailcall:
                print(self._str+name+"(): return "+sym+" = "+repr(val))
        self += 1 #immediate call... 
        self += -1 #... and return
        return val

debug = Debug() # singleton

# Fake class to make use of hasconection in canextend easier...
class AlwaysMatch:
    def __eq__(self,other): return True
    def __repr__(self): return "Always"

# Methods for chopping out similar characters at the start/end
# of a group of words.
# Returns new_prefix,did_succeed
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

# This groups words together based on common prefixes/postfixes,
# aiming to capture several groups of words with the most in common.
# Takes a list of strings; returns a list of Switches.
def filter_list(wordlist_left):
    if (len(wordlist_left) <= 1 or
        (len(wordlist_left) == 2 and '' in wordlist_left)):
        return [Switch(e) for e in wordlist_left]
    # Strangely enough, if this changes to a regular sort algorithm,
    # (i.e. no lambda), this fails to group "for" with "foreach" and
    # "forindex" when running with --special=keywords
    #
    # Doing it this way pakes sure smaller words acting as whole
    # prefixies/postfixes get picked up first, and take longer words too.
    # Also sort by regular cmp(str, str) to assure a fixed order.
    wordlist_left.sort(sort_pred)
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
    max_size = len(wordlist_left[-1])
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
            matches = [[base],[base]]
            #if debug.EXTRA_DEBUG: debug.call("sub_find_matches", locals(), "j", "base", "_pre", "_post")
            for k in range(j+1,len(wordlist_left)):
                e = wordlist_left[k]
                assert len(e) >= i
                pre,post = get_ends(e, i)
                #if debug.EXTRA_DEBUG: debug.dump(locals(), "pre", "post")
                if pre == _pre: matches[0].append(e)
                if post == _post: matches[1].append(e)
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
                    existing[0].extend(match)
                    match = existing[0]
                prefixes[_pre] = match
                if debug.EXTRA_DEBUG: debug.dump(locals(), "_pre", "match")
            else:
                match = matches[1]
                for e in match:
                    wordlist_left.remove(e)
                if existing[1]:
                    del postfixes[post]
                    existing[1].extend(match)
                    match = existing[1]
                postfixes[_post] = match
                if debug.EXTRA_DEBUG: debug.dump(locals(), "_post", "match")
            #if debug.EXTRA_DEBUG: debug.dump(locals(), "prefixes", "postfixes", "children")
            if not match in children: children.append(match)
        #if debug.EXTRA_DEBUG: debug.ret(children)
        if not wordlist_left: break
    else: children += ([e] for e in wordlist_left)
    debug.dump(locals(), "children", "prefixes", "postfixes")
    # Convert all lists of similar words to switches and potentially optimize
    children = [Switch(childL) for childL in children]
    debug.tailcall()
    return optimize(children)

# Configuration (note: metachars need to be escaped):
class regex:
    # Standard grouping: (word1|word2)
    pre = R"\("
    sep = R"\|"
    post = R"\)"
    # Character classes: [gs]
    pre2 = R"\["
    post2 = R"\]"
    sep2 = "" # dummy
    # Make the last group or char optional:
    opt = R"\?"

# Main class: represents a common prefix and postfix with options
# for in-between stuff as "children".
#
# Members:
#    prefix: str or Switch
#    children: list of Switches
#    postfix: str or Switch
class Switch:
    def __init__(self, word_list=None):
        # Shortcuts:
        if word_list is None: return
        if type(word_list) == str: word_list = (word_list,)
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
        # Sort: shortest first with fixed order
        word_list.sort(sort_pred)
        debug.call("Switch.__init__", locals(), "word_list")
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
        if not pre and not post: return j
        skip = 0
        slash = False
        if self.issimple(): simple = True
        elif j.find(regex.sep) is None: simple = True
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
           and type(self.prefix) == str and type(self.postfix) == str
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
            middle = regex.pre+children[0].join()+regex.post
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
        if middle is None:
            middle = regex.pre+regex.sep.join(e.join() for e in children)+regex.post
        #return debug.ret(prefix+middle+postfix)
        if type(self.prefix) == str and type(self.postfix) == str:
            return self.prefix+middle+self.postfix
        return self.postfix.joinwpre_post(self.prefix.joinwpre_post(str(), middle), str())
    def __repr__(self):
        if self.isopt(): return '/opt'
        elif self.issimple(): return repr(self.prefix)
        else:
            if type(self.prefix) == str or not self.prefix.issimple():
                prefix = repr(self.prefix)
            else:
                prefix = repr(self.prefix.prefix) # simple Switch() object
            if type(self.postfix) == str or not self.postfix.issimple():
                postfix = repr(self.postfix)
            else:
                postfix = repr(self.postfix.prefix) # simple Switch() object
            return prefix+"+{"+", ".join([repr(e) for e in self.children])+"}+"+postfix
    def __eq__(self, other):
        return self.__dict__.__eq__(other.__dict__)
    def isopt(self):
        return (type(self.prefix) == str
                and len(self.prefix) == 0
                and not len(self.children)
                and not len(self.postfix))
    def issingle(self):
        return (type(self.prefix) == str
                and len(self.prefix) == 1
                and not len(self.children)
                and not len(self.postfix))
    def issimple(self):
        return (type(self.prefix) == str
                and not len(self.children)
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

# This optimizes a list of children by finding common sets of
# prefixes, children lists, and postfixes (collectively "nodes")
# such that all permutations of such are found as Switches
# (aka "connections"). The children are fused together and the
# pre-/postfixes are made into a Switch().
#
# Note that this has very limited results, including none on
# --special=builtin_functions even. Most common cases that
# are optimized occur when there's an easy catch like:
#   compile rotate uncompile unrotate
# (output would be: "(un)?(compile|rotate)"), so basically
# two or more side-by-side Switches.
def optimize(children):
    global debug
    if not OPTIMIZE or len(children) <= 1:
        return debug.cancelcall("optimize", locals(), "children")
    for i in children:
        if not i.issimple(): break
    else: 
        return debug.cancelcall("optimize", locals(), "children")
    debug.call("optimize", locals(), "children")
    optimizeur = ConnectionMap(children)
    lits = optimizeur.split()
    children2 = lits[0].connections
    optimized = list()
    for e in lits[1:]:
        #assert len(e) > 1
        element = Switch()
        element.children = list()
        for i in e.children: element.children.extend(i)
        element.prefix  = Switch([i for i in e.prefixes if type(i) == str])
        element.postfix = Switch([i for i in e.postfixes if type(i) == str])
        element.prefix.children  += [i for i in e.prefixes if type(i) != str]
        element.postfix.children += [i for i in e.postfixes if type(i) != str]
        debug.dump(locals(), "element")
        optimized.append(element)
    #if optimized: children2 += optimize(optimized)
    if optimized: children2.extend(optimized)
    debug.dump(locals(), "optimized")
    return debug.ret(children2)

class ReferencedSet(object):
    def __init__(self, copy=None):
        self.hashable_refs = dict(copy.hashable_refs) if copy else dict()
        self.other_refs = [
          list(copy.other_refs[0]) if copy else list(),
          list(copy.other_refs[1]) if copy else list()]
    def add(self, reference):
        try:
            hash(reference)
            try:
                self.hashable_refs[reference] += 1
            except KeyError:
                self.hashable_refs[reference] = 1
        except TypeError:
            try:
                index = self.other_refs[0].index(reference)
                (self.other_refs[1])[index] += 1
            except ValueError:
                self.other_refs[0].append(reference)
                self.other_refs[1].append(1)
        return self
    def remove(self, reference):
        try:
            hash(reference)
            self.hashable_refs[reference] -= 1
            if not self.hashable_refs[reference]:
                del self.hashable_refs[reference]
        except TypeError:
            index = self.other_refs[0].index(reference)
            self.other_refs[1][index] -= 1
            if not self.other_refs[1][index]:
                self.other_refs[0].pop(index)
                self.other_refs[1].pop(index)
        return self
    def references(self, reference):
        try:
            hash(reference)
            try:
                return self.hashable_refs[reference]
            except KeyError: return 0
        except TypeError:
            try:
                index = self.other_refs[0].index(reference)
                return self.other_refs[1][index]
            except ValueError: return 0
    def getreferenced(self):
        referenced = zip(*self.other_refs)
        referenced += ([key, self.hashable_refs[key]] for key in self.hashable_refs)
        if not referenced: return referenced
        referenced.sort(key=lambda a: a[1])
        return zip(*referenced)[0]
    __iadd__ = add
    __isub__ = remove
    __delitem__ = remove
    __getitem__ = references
    def __iter__(self): return iter(self.getreferenced())
    def __len__(self): return len(self.other_refs[0])+len(self.hashable_refs)
    def __repr__(self):
        referenced = zip(*self.other_refs)
        referenced += ([key, self.hashable_refs[key]] for key in self.hashable_refs)
        referenced.sort(key=lambda a: a[1])
        return "{"+", ".join([repr(a[0])+": "+repr(a[1]) for a in referenced])+"}"

# Implements the "nodes" and "connection" model and corresponding
# contiguous-region-finding-algorithm required by optimize().
# Relies on ReferencedSet for abstracting pairs of nodes with references.
class ConnectionMap(object):
    def __init__(self, copy=None):
        assert type(self) == ConnectionMap
        if type(copy) == ConnectionMap:
            use_copy = True
        else: use_copy = False
        self.connections = list(copy.connections) if use_copy else list()
        self.children = ReferencedSet(copy.children) if use_copy else ReferencedSet()
        self.prefixes = ReferencedSet(copy.prefixes) if use_copy else ReferencedSet()
        self.postfixes = ReferencedSet(copy.postfixes) if use_copy else ReferencedSet()
        if copy and not use_copy:
            for i in copy: self.addconnection(i)
    def addconnection(self, connection):
        assert connection not in self.connections
        self.connections.append(connection)
        self.children += connection.children
        self.prefixes += connection.prefix
        self.postfixes += connection.postfix
    def makeconnection(self, children, prefix, postfix):
        test = Switch()
        test.children = children
        test.prefix   = prefix
        test.postfix  = postfix
        return test
    def hasconnection(self, *arg):
        test = self.makeconnection(*arg)
        return test in self.connections
    def removeconnection(self, connection):
        assert connection in self.connections
        self.connections.remove(connection)
        self.children -= connection.children
        self.prefixes -= connection.prefix
        self.postfixes -= connection.postfix
    def getconnections(self):
        connections = [
            [connection,
             self.prefixes[connection.prefix]+
             self.postfixes[connection.prefix]+
             self.children[connection.children]
            ] for connection in self.connections
        ]
        connections.sort(key=lambda a: -a[1])
        return zip(*connections)[0]
    def getnodes(self):
        nodes = [
          (None,i,self.children[i])
          for i in self.children
        ]+[
          (True,i,self.prefixes[i])
          for i in self.prefixes
        ]+[
          (False,i,self.postfixes[i])
          for i in self.postfixes
        ]
        nodes.sort(key=lambda a: -a[2])
        return nodes
    def canextend(self, other, current):
        other = dict(other)
        other[current[0]] = current[1:]
        for i in (None,True,False):
            if not i in other:
                other[i] = (AlwaysMatch(),-1)
        if len(other) == 3:
            return self.hasconnection(other[None][0], other[True][0], other[False][0])
        else: raise ValueError()
    def test_addition(self, matches, test):
        if debug.EXTRA_DEBUG: debug.call("ConnectionMap.test_addition", locals(), "matches", "test")
        others = [i for i in (None,True,False) if i != test[0]]
        mapping = {None:0,True:1,False:2}
        curr = [None,None,None]
        curr[mapping[test[0]]] = test[1]
        for i in matches[others[0]]:
            curr[mapping[others[0]]] = i
            for j in matches[others[1]]:
                curr[mapping[others[1]]] = j
                if debug.EXTRA_DEBUG: debug.dump(locals(), "curr")
                if not self.hasconnection(*curr):
                    if debug.EXTRA_DEBUG: return debug.ret(False)
                    else: return False
        if debug.EXTRA_DEBUG: return debug.ret(True)
        else: return True
    # Grow algorithm to find contiguous regions; starts with the nodes
    # that are most referenced, since they are most likely to form
    # such a region.
    def split(self):
        result = [ConnectionMap(self)] # discarded "other/rest" elements
        disc = result[0]
        if not len(self.connections): return result
        global debug
        debug.call("ConnectionMap.split", locals(), "self")
        assert disc.connections[0] in disc.connections
        assert disc.hasconnection(disc.connections[0].children,disc.connections[0].prefix, disc.connections[0].postfix)
        debug.dump(locals(), "disc")
        # Mapping:
        #  None: element.children
        #  True: element.prefix
        #  False: element.postfix
        nodes = disc.getnodes()
        matches = {None:None,True:None,False:None}
        refs_left = dict(matches)
        _i = 0
        while _i < len(nodes)-3:
            initial = dict() # mapping_val: value
            rest = nodes[_i:]
            if debug.EXTRA_DEBUG: debug.dump(locals(), "rest")
            if rest[0][2] == 1: break # nothing worthwhile left; all single referenced
            for i in rest:
                if i[0] in initial: continue
                if not initial:
                    initial[i[0]] = i[1:]
                elif disc.canextend(initial, i):
                    initial[i[0]] = i[1:]
                if len(initial) == 3: break
            else:
                if debug.EXTRA_DEBUG: debug.prnt("not found continue!")
                _i += 1; continue # couldn't find any matches; they are at < _i

            if debug.EXTRA_DEBUG: debug.dump(locals(), "initial")
            for k in initial:
                refs_left[k] = [ initial[k][1] ]
                matches[k]   = [ initial[k][0] ]
            
            if refs_left[None]+refs_left[True]+refs_left[False] == 3:
                _i += 1; continue # nothing worthwhile left

            _break = False
            for i in rest:
                if i[1] == initial[i[0]][0]:
                    continue
                if disc.test_addition(matches, i):
                    matches[i[0]].append(i[1])
                    for refL in refs_left.values():
                        for j in range(len(refL)):
                            refL[j] -= 1
                            if not refL[j]:
                                if debug.EXTRA_DEBUG: debug.prnt("referenced break!")
                                _break = True; break
                        if _break: break
                    if _break: break

            if (sum(( len(matches[k]) for k in matches )) == 3
                or (len(matches[False]) + len(matches[None]) == 2
                    and not matches[False][0] and not matches[None][0])):
                _i += 1; continue

            debug.dump(locals(), "matches")
            new = ConnectionMap()
            for children in matches[None]:
                for prefix in matches[True]:
                    for postfix in matches[False]:
                        i = self.makeconnection(
                          children, prefix, postfix
                        )
                        disc.removeconnection(i)
                        new.addconnection(i)
            result.append(new)
            debug.dump(locals(), "new")
            nodes = disc.getnodes()
        return debug.ret(result)
    def __repr__(self): return repr(self.__dict__)

# Swaps escaping on metachars, so input can be regarded literally
def swap_escapes(char, string):
    spilt = string.split("\\"+char)
    for i in range(len(spilt)):
        spilt[i] = spilt[i].replace(char, "\\"+char)
    return char.join(spilt)

def hasopt(arg, longn, shortn, argument=False):
    if shortn is not None:
        shortn = "-"+shortn
    if not argument:
        if longn is not None:
            longn = "--"+longn
        end = arg.index("--") if "--" in arg else float("inf")
        if (longn is not None and longn in arg
           and arg.index(longn) < end):
            ret = True
        elif (shortn is not None and shortn in arg
           and arg.index(shortn) < end):
            ret = True
        else: ret = False
        if ret:
            try:
                while True:
                    if arg.index(longn) < end:
                        arg.remove(longn)
                    else: break
            except ValueError: pass
            except TypeError: pass
            try:
                while True:
                    if arg.index(shortn) < end:
                        arg.remove(shortn)
                    else: break
            except ValueError: pass
            except TypeError: pass
        return ret
    ret = list()
    if longn is not None:
        longn = "--"+longn+"="
        len_longn = len(longn)
    else:
        len_longn = float("inf")
    was_short = False
    for i in arg[:]:
        if was_short:
            was_short = False; continue
        elif i == "--": break
        elif len(i) > len_longn and i[:len(longn)] == longn:
            ret.append(i[len(longn):])
            arg.remove(i)
        elif i == shortn:
            idx = arg.index(i)
            arg.remove(i)
            ret.append(arg.pop(idx))
            was_short = True
    return ret

def main(arg=sys.argv[1:]):
    global debug
    PRINT_RESULT = False
    XML_ESCAPE = False
    ECHO = False
    if hasopt(arg, "EXTRA_DEBUG", "e"):
        debug.DEBUG = True
        debug.EXTRA_DEBUG = True
    if hasopt(arg, "DEBUG", "d"):
        debug.DEBUG = True
    if hasopt(arg, "PRINT_RESULT", "r"):
        PRINT_RESULT = True
    if hasopt(arg, "XML_ESCAPE", "x"):
        XML_ESCAPE = True
    if hasopt(arg, "ECHO", None):
        ECHO = True
    if hasopt(arg, "OPTIMIZE", "o"):
        global OPTIMIZE
        OPTIMIZE = True
    special = hasopt(arg, "special", "s", True)
    filelist = hasopt(arg, "file", "f", True)
    if hasopt(arg, None, ""): filelist.append("-")
    
    word_list = arg[:]
    if "--" in word_list:
        word_list.remove("--")
    
    for s in special:
        word_list += {
            "builtin_functions": builtin_functions,
            "keywords": keywords,
            "operators": operators
        }[s]
    
    if filelist or not word_list:
        for line in fileinput.input(filelist):
            line = line.rstrip() # strip trailing spaces and \n
            if ECHO and line: print(line)
            if line: word_list.append(line) # ignore blank lines
    word_list = [i.replace("\\", "\\\\") for i in set(word_list)] # protect against duplicates, escape backslashes
    
    result = filter_list(word_list)
    if PRINT_RESULT: print(repr(result))
    result = regex.sep.join(sw.join() for sw in result)
    
    for char in (".","(","|",")","?","[","]","+","*","{","}"):
        result = swap_escapes(char, result)
    if XML_ESCAPE:
        result = result.replace("&", "&amp;")
        result = result.replace("<", "&lt;")
        result = result.replace(">", "&gt;")
    
    return result

if __name__ == "__main__":
    print(main())
