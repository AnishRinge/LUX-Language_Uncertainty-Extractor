import re
q='## 1. Project objective and completed scope'
for p in [r'^## (\\d+)\\. (.*)$',r'^## (\d+)\. (.*)$',r'^## (\d+)\. (.*)$']:
 print(repr(p),re.match(p,q))
