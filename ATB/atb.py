from typing import Optional, Any, List, Dict, Tuple, Union
import itertools
from collections import defaultdict, namedtuple
import re


def is_chinese_char(c: str) -> bool:
    '''
    Check if a character falls within Chinese unicode values
    
    note: combined code pages:
    0x2a700 - 0x2b73f, 0x2b740 - 0x2b81f, 0x2b820 - 0x2ceaf
    '''
    n = ord(c)
    return (0x3400  <= n <= 0x4dbf or
            0x4e00  <= n <= 0x9fff or
            0xf900  <= n <= 0xfaff or
            0x20000 <= n <= 0x2a6df or
            0x2a700 <= n <= 0x2ceaf or
            0x2f800 <= n <= 0x2fa1f) 

def no_validator(c: str) -> bool:
    return True

class Graph():
    """Simple graph obj to work with Dijkstra's Algorithm"""
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}
        
    def add_edge(self, from_node, to_node, weight):
        """Add an edge to the graph in one direction: from_node -> to_node """
        self.edges[from_node].append(to_node)
        self.weights[(from_node, to_node)] = weight
        
    def dijkstra(self, initial: Any, end: Any) -> Optional[list[Any]]:
        shortest_paths = {initial: (None, 0)}
        current_node = initial
        visited = set()
        
        while current_node != end:
            visited.add(current_node)
            destinations = self.edges[current_node]
            weight_to_current = shortest_paths[current_node][1]
            
            for next_node in destinations:
                weight = self.weights[(current_node, next_node)] + weight_to_current
                if next_node not in shortest_paths:
                    shortest_paths[next_node] = (current_node, weight)
                else:
                    current_shortest_weight = shortest_paths[next_node][1]
                    if current_shortest_weight > weight:
                        shortest_paths[next_node] = (current_node, weight)
            next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
            
            if not next_destinations:
                return None
            
            # next node is the destination with the lowest weight
            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
            
        # work back through destinations in shortest path
        path = []
        while current_node is not None:
            path.append(current_node)
            next_node = shortest_paths[current_node][0]
            current_node = next_node
            
        # spin back flip it and reverse it
        return path[::-1]


class Node():
    """One Node of a Trie"""

    def __init__(self) -> None:
        self.children: Dict[str, Node] = {}
        self.value: Optional[Any] = None


class ATB():
    """Trie structure, with functions for insertion, retrieval, and deletion."""

    def __init__(self):
        self.root = Node()

    def search(self, key: str) -> Optional[Any]:
        """Alias for find() function.
        Find `key` in alphadict; return `key`'s value"""
        return self.find(key)

    def find(self, key: str) -> Optional[Any]:
        '''Find `key` in alphadict; return `key`'s value.'''
        node = self.root
        for char in key:
            if char in node.children:
                node = node.children[char]
            else:
                return None
        return node.value

    def insert(self, key: str, value: Any) -> None:
        """Insert key/value pair into trie, iteratively."""
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = Node()
            node = node.children[char]

        node.value = value
    
    def delete(self, key: str) -> bool:
        '''delete `key` from the trie, recursively, if `key` exists in trie.'''

        def _delete(node: Node, key: str, d: int) -> bool:
            '''clear the node corresponding to key[d], and delete the child 
            key[d+1] if that subtrie is completely empty, and return whether 
            `node` has been cleared.'''
            if d == len(key):
                node.value = None
            else:
                c = key[d]
                if c in node.children and _delete(node.children[c], key, d+1):
                    del node.children[c]

            return node.value is None and len(node.children) == 0

        return _delete(self.root, key, 0)

    def keys_with_prefix(self, prefix: str) -> List[str]:
        '''Check to see if string `prefix` is a prefix for any existing words in the
        trie. If it is, return a list of all words in the trie that start with `prefix`.
        Beware using this function on single characters; the resulting list can be 
        massive and slow down the overall program significantly.'''

        def _collect(x: Optional[Node], prefix: List[str], results: List[str]) -> None:
            """append keys under node `x` matching the given prefix to `results`."""
            if x is None:
                return
            if x.value is not None:
                prefix_str = "".join(prefix)
                results.append(prefix_str)
            for c in x.children:
                prefix.append(c)
                _collect(x.children[c], prefix, results)
                del prefix[-1]

        def _get_node(node: Node, key: str) -> Optional[Node]:
            '''Find a node using `key`'''
            for char in key:
                if char in node.children:
                    node = node.children[char]

                else:
                    return None
            return node

        root = self.root
        results: List[str] = []
        x = _get_node(root, prefix)
        _collect(x, list(prefix), results)

        return results
    
        
    def get_matches(self, s: str) -> List[Any]:
        '''Search alphaDict dictionary tree for all possible word matches in string s.
        
        *Previous version of get_matches that couldn't handle numbers in a string.*
        
        Returns a list composed of tuples of the matching term, and the index range in 
        `s` where that term was found, e.g. (term, range(start_index, stop_index))'''

        sets = []
        all_words = []

        for i in range(len(s)):
            for j in range(i+1, len(s)+1):
                proto = s[i:j]                   # make a substring
                result = self.find(proto)        # check if it's in the trie
                prefix = self.keys_with_prefix(  # check if it's the prefix to something
                    proto) if len(proto) > 1 else [proto]  # if it's at least 2 chars long

                if result:                       # if it's a word in the trie
                                                 # put the potential match in sets
                    sets.append([proto, range(i, j)])
                else:
                    if prefix:                   # otherwise if it's part of a trie branch
                        continue
                    else:
                        all_words.append(sets)   # else append the sets to all words
                        sets = []                # and clear working sets
                        break
        # handle unappended sets
        if sets:
            all_words.append(sets)

        return all_words
    
    def _flatten(self, s: str) -> List[List[str]]:
        """
        Segment string `s` and return a flat list of matches. This method uses non-python-specific code.
        """
        def _flatten(x): # yes this is python specific, but whatever, it's translateable
            return sum(map(_flatten, x), []) if isinstance(x, tuple) or isinstance(x, list) else [x]
        def _pair(x):
            return [[x[i], x[i+1]] for i in range(0, len(x), 2)]
        
        return _pair(_flatten(self.get_matches(s)))

    def greedy_segment(self, s: str) -> List[str]:
        """
        Segment string `s` into a list of individual words, greedily.
        """
        
        s = re.sub("[\W]", "", s)
        
        i = 0
        ret = []
        
        wds = self._flatten(s)
        
        while i < len(wds):
            group = [ x for x in wds if x[-1][0] == i ]  # find the group of all words that start at index i
            if group:
                item = max(group, key=lambda x: len(x[1])) # find the longest one
                ret.append(item[0])
                i += len(item[1]) # increase i by its length
            else:
                i += 1
                
        return ret
                
    
    def dijkstra_segment(self, s:str) -> List[str]:
        '''Use Dijkstra's algorithm to traverse a weighted graph of the dictionary matches in string `s`.
        This method prefers word length, minimizing the number of single character words in a segmented sentence.'''
        
        g = Graph()
        Match = namedtuple('Match', ['start', 'end', 'length', 'weight']) # convenience class
        
        def _parse_match(m): # convenience function
            """return word start_index, end_index, length, and calculated edge weight (1 / len^2)"""
            return m[1][0], m[1][-1]+1, len(m[1]), 1/(len(m[0]) ** 2)
        
        # parse out all matches in the sentence
        matches = [Match(*_parse_match(match)) for match in self._flatten(s+'z')]
        
        # populate the graph with edges
        for ind, match in enumerate(matches): # starting with match
            for other in matches[ind + 1:]: # iterate over all forward matches
                if other.start == match.end: # if the match ends where the forward one starts
                    g.add_edge(match.start, other.start, match.weight)
                    
        # try to make a path from 0 to the char at the end, decrementing until a path exists
        x = len(s)
        while not g.dijkstra(0, x) and x > 0:
            x -= 1
        path = g.dijkstra(0, x)[1:]
        
        # if we didn't end up making a path then our dijkstra failed
        if not path:
            raise Exception("Something went wrong with pathfinding")
        
        ## assemble segments into a list
        segments = []
        
        last = 0
        for curr in path:
            segments.append(s[last:curr])
            last = curr
        
        return segments