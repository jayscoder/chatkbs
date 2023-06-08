import asyncio
import concurrent.futures


def process_file(args):
    name, = args

    for i in range(10):
        print(f'{name}: {i}')
        yield i


# 并发执行任务
with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    arguments = [(f't{idx}',) for idx, _ in enumerate([1, 2, 3, 4, 5])]

    results = executor.map(process_file, arguments)
    for result in results:
        print(result)


