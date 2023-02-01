import numpy as np
import edcl as di

ref = di.load_pickle('comp_ref.pickle')
spec = di.load_pickle('comp_spec.pickle')

print(np.isnan(ref).any())
print(ref[-1])
print(spec[-1])

print(np.shape(ref))
print(np.shape(spec))

x = np.searchsorted(ref, spec)
y = list()
for value in spec:
    y.append(np.searchsorted(ref, value))
y = np.array(y)

print(x-y)
print(x)
print(y)


print('\n\n\n\n\n')
print(f'For value: {spec[19]}')
print(ref[899])
print(ref[898])
print(ref[436])

print(ref)


print(ref - np.sort(ref))

