# coding: utf8
import sys # for use via CLI
import fileinput # simple inputting of files and stdin
                 # (used when we don't have a wordlist
                 # and for each --file=/-f argument).

#str = unicode

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

optimizeable = [
  "compress", "uncompress",
  "optimize", "unoptimize"
]

complex_optimizeable = [
  "iok",  "iuk",  "iyk",  "itk",
  "ioki", "iuki", "iyki", "itki"
]

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
        self._str = ''
        self._indent = "\t"
    def change(self, val):
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
                comp += ("; " if comp != " " else '') + s + " = " + repr(local[s])
        else: comp = ''
        print(self._str+name+"():"+comp)
        self.change(1)
    # Print a message with the current indent
    def prnt(self, string):
        if not self.DEBUG: return
        print(self._str+str(string))
    # Dump local symbols
    def dump(self, local, *sym):
        if not self.DEBUG: return
        comp = ''
        if not sym: sym = local.keys()
        for s in sym:
            assert type(s) == str
            comp += ("; " if comp else '') + s + " = " + repr(local[s])
        print(self._str+comp)
    # Print a return value; passes value back for syntax like:
    #   return debug.ret(...)
    def ret(self, val):
        if not self.DEBUG: return val
        self.change(-1)
        print(self._str+"return "+repr(val))
        return val
    # Set up the next call as a tail call, e.g.:
    #   debug.tailcall()
    #   return fn(...)
    def tailcall(self):
        if not self.DEBUG: return
        self.change(-1)
        self._str += "return "
    # In case of a degenerate case or such after a tail call,
    # this makes sure that the output indicates what was returned
    # without full output of arguments and values.
    #   val = local[sym] if sym is not None else val = local
    # Returns the "val"; doesn't print anything if not following
    # a declared tail call.
    def cancelcall(self, name, local, sym=None):
        if sym is None:
            val = local
        else:
            val = local[sym]
        if self.DEBUG:
            if self._str[-7:] == "return ":
                print(self._str+name+"(): return "+(sym+" = " if sym else '')+repr(val))
            self.change(1) #immediate call... 
            self.change(-1) #... and return
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
def get_ends(string, sz):
    return string[:sz],string[-sz:]

# Configuration (note: metachars need to be escaped):
class regex:
    # Standard grouping: (word1|word2)
    pre = R"\("
    sep = R"\|"
    post = R"\)"
    # Character classes: [gs]
    pre2 = R"\["
    post2 = R"\]"
    sep2 = '' # dummy
    # Make the last group, char class, or char optional:
    opt = R"\?"



# This groups words together based on common prefixes/postfixes,
# aiming to capture several groups of words with the most in common.
# Takes a list of strings; returns a list of Switches.
def filter_list(wordlist_left):
    if len(wordlist_left) <= 1:
        return [Switch(e) for e in wordlist_left]
    wordlist_left.sort(sort_pred)
    if not wordlist_left[0]:
        wordlist_left.pop(0)
        children = filter_list(wordlist_left)
        children.append(Switch(''))
        return children
    global debug
    debug.call("filter_list", locals(), "wordlist_left")
    children = list()
    prefixes = dict()
    postfixes = dict()
    # Get a starting estimate: we work backwards from max_size-1 through 0
    # to get the longest possible prefix or postfix (the one with more
    # matches is used).
    max_size = len(wordlist_left[-1])
    # But if only using the first character, we actually only want 1:
    existing = [None,None]
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
            # If there's already a similar prefix (i.e. this is a prefix
            # of the prefix, or sub-prefix), add this to its list, as this
            # current prefix's items will be handled later in recursion
            try: existing[0] = prefixes[_pre[0]]
            except KeyError: existing[0] = None
            try: existing[1] = postfixes[_post[-1]]
            except KeyError: existing[1] = None

            if len(matches[0]) + len(matches[1]) == 2:
                j += 1; continue
            # Take the one with more items
            elif len(matches[0]) >= len(matches[1]):
                match = matches[0]
                for e in match:
                    wordlist_left.remove(e)
                if existing[0]:
                    existing[0].extend(match)
                    match = existing[0]
                else:
                    prefixes[_pre[0]] = match
                    children.append(match)
                if debug.EXTRA_DEBUG: debug.dump(locals(), "_pre", "match")
            else:
                match = matches[1]
                for e in match:
                    wordlist_left.remove(e)
                if existing[1]:
                    existing[1].extend(match)
                    match = existing[1]
                else:
                    postfixes[_post[-1]] = match
                    children.append(match)
                if debug.EXTRA_DEBUG: debug.dump(locals(), "_post", "match")
            #if debug.EXTRA_DEBUG: debug.dump(locals(), "prefixes", "postfixes", "children")
        #if debug.EXTRA_DEBUG: debug.ret(children)
        if not wordlist_left: break
    else: children.extend( ([e] for e in wordlist_left) )
    debug.dump(locals(), "children", "prefixes", "postfixes")
    # Convert all lists of similar words to switches and potentially optimize
    children = [Switch(childL) for childL in children]
    debug.tailcall()
    return optimize(children)

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
    other_connection_maps = optimizeur.split()
    optimized = list()
    for e in other_connection_maps:
        #assert len(e) > 1
        element = Switch()
        element.children = [k for i in e.children for k in i]
        if OPTIMIZE == -1: element.children = optimize(element.children)
        if len(e.prefixes) > 1:
            element.prefix  = Switch([i for i in e.prefixes if type(i) == str])
            element.prefix.children.extend( (i for i in e.prefixes if type(i) != str) )
            if OPTIMIZE == -1: element.prefix.children = optimize(element.prefix.children)
        else: element.prefix = list(e.prefixes)[0]
        if len(e.postfixes) > 1:
            element.postfix = Switch([i for i in e.postfixes if type(i) == str])
            element.postfix.children.extend((i for i in e.postfixes if type(i) != str))
            if OPTIMIZE == -1: element.postfix.children = optimize(element.postfix.children)
        else: element.postfix = list(e.postfixes)[0]
        if debug.EXTRA_DEBUG: debug.dump(locals(), "element")
        optimized.append(element)
    if optimized:
        debug.dump(locals(), "optimized")
        if OPTIMIZE == -1:
            children2 = optimize(optimized)
        else:
            children2 = optimized
        children2.extend(optimizeur.connections) # what is left after making other groups of connections
    else: children2 = optimizeur.connections
    return debug.ret(children2)

# Main class: represents a common prefix and postfix with options
# for in-between stuff as "children".
#
# Members:
#    prefix: str or Switch
#    children: list of Switches
#    postfix: str or Switch
class Switch(object):
    EXPAND_SINGLES=False
    def __init__(self, word_list=None):
        # Shortcuts:
        if word_list is None: return
        if type(word_list) == str: word_list = (word_list,)
        if len(word_list) <= 1:
            self.prefix = word_list[0] if word_list else ''
            self.children = list()
            self.postfix = ''
            return
        # Else we take the long way:
        global debug
        # Sort: shortest first with fixed order
        word_list.sort(sort_pred)
        debug.call("Switch.__init__", locals(), "word_list")
        smallest = word_list[0]
        middle = list(word_list)
        
        # Find a common prefix and common postfix
        prefix = ''
        while len(prefix) < len(smallest):
            (prefix,s) = similar(False, middle, prefix)
            if not s: break
        
        postfix = ''
        while len(prefix)+len(postfix) < len(smallest):
            (postfix,s) = similar(True, middle, postfix)
            if not s: break
        debug.dump(locals(), "middle")
        
        self.prefix = prefix
        self.postfix = postfix
        # And return to filter_list to sort through what's left
        if len(prefix) or len(postfix):
            self.children = filter_list(middle)
            # If we aggresively optimized down to one child, merge it with ourselves
            if len(self.children) == 1:
                child = self.children[0]
                parent = child
                while type(parent.prefix) == Switch:
                    parent = parent.prefix
                parent.prefix = prefix + parent.prefix
                parent = child
                while type(parent.postfix) == Switch:
                    parent = parent.postfix
                parent.postfix = parent.postfix + postfix
                self.prefix = child.prefix
                self.postfix = child.postfix
                self.children = child.children
        else: self.children = [Switch(m) for m in middle]
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
        if self.isopt(): return ''
        #debug.call("Switch.join", locals(), "self")
        children = self.children
        # For simple lists like c(ompile|aller) or (ha|po)t,
        # add the prefix or postfix to each one instead of
        # using grouping
        if (self.EXPAND_SINGLES and len(children) > 1
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
            middle = ''
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
        if type(self.prefix) == str:
            ret = self.prefix+middle
        elif self.prefix.issimple():
            ret = self.prefix.prefix+middle
        else:
            ret = self.prefix.joinwpre_post('', middle)
        if type(self.postfix) == str:
            ret += self.postfix
        elif self.postfix.issimple():
            ret += self.postfix.prefix
        else:
            ret = self.postfix.joinwpre_post(ret, '')
        return ret
    def permute(self):
        if self.issimple(): return [self.prefix]
        result = list()
        if type(self.prefix) == str:
            prefixes = [self.prefix]
        else: prefixes = self.prefix.permute()
        if type(self.postfix) == str:
            postfixes = [self.postfix]
        else: postfixes = self.postfix.permute()
        
        for j in self.children:
            j = j.permute()
            result.extend((pre+mid+post
                           for pre in prefixes
                           for mid in j
                           for post in postfixes))
        return result
    def whole_match(self, string):
        if self.issimple(): return (self.prefix == string)
        start = 0; end = 0
        if type(self.prefix) == str:
            start = len(self.prefix)
            if len(string) < start: return False
            if string[:start] != self.prefix: return False
        else:
            start = self.prefix.match(string, 1)
            if not start: return False
        string = string[start:]
        if type(self.postfix) == str:
            end = -len(self.postfix)
            if len(string) < -end: return False
            if string[:end] != self.postfix: return False
        else:
            end = self.postfix.match(string, -1)
            if not end: return False
        string = string[:end]
        for ch in children:
            if ch.whole_match(string): break
        else: return False
        return True
    def match(self, string, direction=0):
        if not direction:
            if self.whole_match(string):
                return len(string)
            else: return None
        if self.issimple():
            if direction > 0:
                return string.startswith(self.prefix) or None
            else:
                return string.endswith(self.prefix) or None
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
        if type(other) == str:
            return self.issimple() and self.prefix == other
        return self.prefix == other.prefix and self.postfix == other.postfix and self.children == other.children
    def isopt(self):
        return (type(self.prefix) == str
                and type(self.postfix) == str
                and len(self.prefix) == 0
                and not len(self.children)
                and not len(self.postfix))
    def issingle(self):
        return (type(self.prefix) == str
                and type(self.postfix) == str
                and len(self.prefix) == 1
                and not len(self.children)
                and not len(self.postfix))
    def issimple(self):
        return (type(self.prefix) == str
                and type(self.postfix) == str
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
    def __hash__(self):
        """Why is this needed?"""
        raise TypeError("non-hashable")

class Parser:
    def __init__(self, string):
        self._i = 0
        self._string = string
        self._sz = len(string)
    def char(self,offset=0):
        i = self._i + offset
        return self._string[i] if i < self._sz else None
    def incr(self,step=1): self._i += step
    def end(self): return self._i >= self._sz
    def shouldexit(self):
        return (self.end() or self.char() in (")","|"))
    def _getswitches(self):
        i = self._i; _str = self._string[i:]
        debug.call("Parser._getswitches", locals(), "i", "_str")
        result = list()
        while not self.end():
            if self.char() == ")": break
            elif result:
                if self.char() == "?":
                    result[-1].children.append(Switch(list()))
                    self.incr()
                    if self.end(): break
                assert self.char() == "|"
                self.incr()
            result.append(self._getswitch())
        return debug.ret(result)
    # This allows read-in of complex Switches
    def _getswitch(self):
        i = self._i; _str = self._string[i:]
        debug.call("Parser._getswitch", locals(), "i", "_str")
        result = None
        lits = list()
        while not self.end():
            lits.append(self._getsimple())
            if self.shouldexit(): break
        if len(lits) == 1: return debug.ret(lits[0])
        result = Switch()
        if len(lits) == 3:
            result.prefix = lits[0]
            assert not lits[1].prefix
            result.children = lits[1].children
            assert not lits[2].prefix
            result.postfix = lits[2]
            result.postfix.prefix = lits[1].postfix
        else:
            assert len(lits) == 2
            result.prefix = Switch(lits[0].prefix)
            result.children = lits[0].children
            assert not lits[1].prefix
            result.postfix = lits[1]
            result.postfix.prefix = lits[0].postfix
        return debug.ret(result)
    def _getsimple(self):
        i = self._i; _str = self._string[i:]
        debug.call("Parser._getsimple", locals(), "i", "_str")
        result = Switch(list())
        while not self.end():
            if (self.char() in ("(",")","|","[")
                or (self.char() != "\\"
                   and self.char(1) == "?")): break
            if self.char() == "\\":
                self.incr()
            result.prefix += self.char()
            self.incr()
        if self.shouldexit(): return debug.ret(result)
        if self.char() == "(":
            self.incr()
            result.children = self._getswitches()
            assert self.char() == ")"
            if self.char(1) == "?":
                result.children.append(Switch(list()))
                assert result.children[-1].isopt()
                self.incr()
            self.incr()
        elif self.char() == "[":
            self.incr()
            while not self.end():
                if self.char() == "]": break
                result.children.append(Switch(self.char()))
                self.incr()
            assert self.char() == "]"
            if self.char(1) == "?":
                result.children.append(Switch(list()))
                assert result.children[-1].isopt()
                self.incr()
            self.incr()
        elif self.char(1) == "?":
            result.children.append(Switch(self.char()))
            result.children.append(Switch(list()))
            self.incr(2)
        else: assert self.end()
        if self.shouldexit(): return debug.ret(result)
        while not self.end():
            if (self.char() in ("(",")","|","[")
                or (self.char() != "\\"
                   and self.char(1) == "?")): break
            if self.char() == "\\":
                self.incr()
            result.postfix += self.char()
            self.incr()
        return debug.ret(result)
    def run(self):
        _str = self._string
        debug.call("Parser.run", locals(), "_str")
        self._i = 0
        debug.tailcall()
        return self._getswitches()

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
    mapping = {None:0,True:1,False:2}
    types = mapping.keys()
    def __init__(self, copy=None):
        assert type(self) == ConnectionMap
        if type(copy) == ConnectionMap:
            self.children  = ReferencedSet(copy.children)
            self.prefixes  = ReferencedSet(copy.prefixes)
            self.postfixes = ReferencedSet(copy.postfixes)
            self.connections = list(copy.connections)
        else:
            self.children  = ReferencedSet()
            self.prefixes  = ReferencedSet()
            self.postfixes = ReferencedSet()
            self.connections = list()
            if copy is not None:
                for i in copy:
                    self.addconnection(i)
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
    def tryextend(self, other, current):
        other[current[0]] = current[1:]
        if len(other) == 1: return True
        added = list()
        for i in self.types:
            if not i in other:
                other[i] = (AlwaysMatch(),-1)
                added.append(i)
        if len(other) == 3:
            val = self.hasconnection(other[None][0], other[True][0], other[False][0])
            for i in added:
                del other[i]
            if val:
                return True
            else:
                del other[current[0]]
                return False
        else: raise ValueError()
    def test_addition(self, matches, test):
        if debug.EXTRA_DEBUG: debug.call("ConnectionMap.test_addition", locals(), "matches", "test")
        mapping = self.mapping
        others = [i for i in self.types if i != test[0]]
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
        result = list()
        if not len(self.connections): return result
        global debug
        debug.call("ConnectionMap.split", locals(), "self")
        # Mapping:
        #  None: element.children
        #  True: element.prefix
        #  False: element.postfix
        nodes = self.getnodes() # updated on changes
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
                self.tryextend(initial, i)
                if len(initial) == 3: break
            else:
                if debug.EXTRA_DEBUG: debug.prnt("not found continue!")
                _i += 1; continue # couldn't find any matches; they are at < _i
            
            for k in initial:
                refs_left[k] = [ initial[k][1]-1 ] # one ref used already
                matches[k]   = [ initial[k][0] ]
            if debug.EXTRA_DEBUG: debug.dump(locals(), "initial", "matches", "refs_left")
            
            if sum((len(i) for i in refs_left.values())) == 0:
                _i += 1; continue # nothing worthwhile left

            _break = False
            if debug.EXTRA_DEBUG: debug.prnt("start iter rest")
            for i in rest:
                if i[1] == initial[i[0]][0]:
                    continue
                # Test if we can add it:
                if self.test_addition(matches, i):
                    matches[i[0]].append(i[1])
                    if debug.EXTRA_DEBUG: debug.dump(locals(), "matches", "refs_left")
                    refs_taken = 1
                    other = (k for k in self.types if k != i[0])
                    for k in other:
                        assert len(matches[k]) == len(refs_left[k])
                        refs_taken *= len(matches[k])
                    refs_left[i[0]].append(i[2]-refs_taken)
                    for k in refs_left:
                        if k == i[0]: continue
                        refL = refs_left[k]
                        for j in range(len(refL)):
                            refL[j] -= 1
                            if not refL[j]:
                                if debug.EXTRA_DEBUG: debug.prnt("referenced break")
                                _break = True; break
                        if _break: break
                elif _break and debug.EXTRA_DEBUG: debug.dump(locals(), "refs_left")
                if _break: break
            if debug.EXTRA_DEBUG: debug.prnt("end iter rest")

            if (sum( (len(matches[k]) for k in matches) ) == 3
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
                        self.removeconnection(i)
                        new.addconnection(i)
            result.append(new)
            debug.dump(locals(), "new")
            nodes = self.getnodes() # it has now changed
        return debug.ret(result)
    def __repr__(self): return repr(self.__dict__)

# Swaps escaping on metachars, so input can be regarded literally
# FIXME: actually doesn't work well with backslashes
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
            try:
                ret.append(arg.pop(idx))
            except IndexError:
                ret.append('')
            was_short = True
    return ret

def main(arg=sys.argv[1:]):
    global debug
    global OPTIMIZE
    PRINT_RESULT = False
    XML_ESCAPE = False
    ECHO = False
    if hasopt(arg, "EXTRA_DEBUG", "D"):
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
        OPTIMIZE = True
    if hasopt(arg, "RE_OPTIMIZE", "O"):
        OPTIMIZE = -1
    if hasopt(arg, "EXPAND_SINGLES", "s"):
        Switch.EXPAND_SINGLES = True
    UNPARSE = hasopt(arg, "UNPARSE", None)
    special = hasopt(arg, "special", "s", True)
    filelist = hasopt(arg, "file", "f", True)
    evals    = hasopt(arg, "eval", "e", True)
    regexoutput = hasopt(arg, "regex-out", None, True)
    if hasopt(arg, None, ''): filelist.append("-")
    
    word_list = arg[:]
    try: word_list.remove("--")
    except ValueError: pass
    
    for s in special:
        word_list.extend({
            "builtin_functions": builtin_functions,
            "keywords": keywords,
            "operators": operators,
            "optimizeable": optimizeable
        }[s])
    for e in evals:
        if ECHO: print(e)
        word_list.extend(eval(e))
    
    if filelist or not word_list:
        for line in fileinput.input(filelist):
            line = line.rstrip() # strip trailing spaces and \n
            if ECHO and line: print(line)
            if line: word_list.append(line) # ignore blank lines
    
    if UNPARSE:
        result = Parser("|".join(word_list)).run()
        if PRINT_RESULT:
            print(repr(result))
        permutations = list()
        for i in result:
            permutations.extend(i.permute())
        return "\n".join(permutations)
    else:
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
        for outfile in regexoutput:
            #with f as open(outfile, 'w'):
            f = open(outfile, 'w')
            if ECHO: print(f)
            f.write(result+"\n")
        return result

if __name__ == "__main__":
    print(main())
