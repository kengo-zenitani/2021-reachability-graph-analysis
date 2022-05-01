import random
from network import Network


random.seed(1)


for i in range(50):
    Network(name=f'nw1-{i + 1}', nv2=2, n0=1, n1=2, c12=3, n2=3, c23=8, n3=24).save()  # paths=48, vuls=28

for i in range(50):
    Network(name=f'nw2-{i + 1}', nv2=3, n0=2, n1=3, c12=3, n2=3, c23=10, n3=30).save()  # paths=180, vuls=48

for i in range(50):
    Network(name=f'nw3-{i + 1}', nv2=3, n0=3, n1=4, c12=3, n2=4, c23=12, n3=48).save()  # paths=432, vuls=60

for i in range(50):
    Network(name=f'nw4-{i + 1}', nv2=4, n0=3, n1=4, c12=5, n2=4, c23=14, n3=56).save()  # paths=840, vuls=88

for i in range(50):
    Network(name=f'nw5-{i + 1}', nv2=4, n0=3, n1=4, c12=5, n2=5, c23=16, n3=80).save()  # paths=960, vuls=104


random.seed(1)


for i in range(50):
    Network(name=f'nw6-{i + 1}', nv2=8, n0=6, n1=8, c12=6, n2=12, c23=48, n3=576).save()  # paths=13824, vuls=576

for i in range(50):
    Network(name=f'nw7-{i + 1}', nv2=12, n0=6, n1=12, c12=8, n2=16, c23=50, n3=800).save()  # paths=28800, vuls=984

for i in range(50):
    Network(name=f'nw8-{i + 1}', nv2=16, n0=8, n1=16, c12=6, n2=24, c23=55, n3=1320).save()  # paths=42240, vuls=1648

for i in range(50):
    Network(name=f'nw9-{i + 1}', nv2=20, n0=8, n1=16, c12=4, n2=32, c23=136, n3=2176).save()  # paths=69632, vuls=2640

for i in range(50):
    Network(name=f'nw10-{i + 1}', nv2=24, n0=8, n1=16, c12=4, n2=64, c23=240, n3=3840).save()  # paths=122880, vuls=4512


random.seed(1)


for hub_count in [0, 8, 16, 24, 32, 40, 48, 56]:
    for i in range(50):
        Network(name=f'nw10-h{hub_count}-{i + 1}', nv2=24, n0=8, n1=16, c12=4, n2=64, c23=240, n3=3840, hub_count=hub_count).save()


