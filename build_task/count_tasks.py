from questions import QUESTION_CATEGORIES

print('=' * 60)
print('Task category statistics')
print('=' * 60)

total = 0
for cat, subcats in QUESTION_CATEGORIES.items():
    print(f"\n[{cat}]")
    cat_total = 0
    for subcat, tasks in subcats.items():
        print(f"  {subcat:6s}: {len(tasks):2d} tasks")
        for t in tasks:
            print(f"           - {t}")
        cat_total += len(tasks)
    print(f"  subtotal: {cat_total:2d} tasks")
    total += cat_total

print('\n' + '=' * 60)
print(f'Total: {total} tasks')
print('=' * 60)

# Calculate the category ratio
info_tasks = sum(len(t) for t in QUESTION_CATEGORIES['graph_info_retrieval'].values())
theory_tasks = sum(len(t) for t in QUESTION_CATEGORIES['graph_theory'].values())
print(f"\nRatio: graph_info_retrieval ({info_tasks}) : graph_theory ({theory_tasks}) = 1 : {theory_tasks/info_tasks:.2f}")
