'''
Function tri_Traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''



def tri_Traversal(cost, heuristic, start_point, goals):
    l = []

    t1=[start_point]
    first=True
    visited=set()

    while t1:
        node=t1.pop()
        if first==True:
            t1.append(start_point)
            first=False
        if node not in visited:
            visited.add(node)
            if node in goals:
                t1.append(node)
                break
            for val in cost[node]:
                if val>0:
                    if cost[node].index(val) in goals:
                        t1.append(cost[node].index(val))
                        break
                    if cost[node].index(val) not in visited:
                        t1.append(cost[node].index(val))

    print(t1)
    # t1 <= dfs traversal
    # t2 <= ucs	
    # t3 <= A_star_Traversal

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l

