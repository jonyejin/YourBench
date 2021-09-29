epsilon = [0, 0.05, 0.1]
accuracies = [0.9, 0.8, 0.7]
results = list(zip(epsilon,accuracies))
print(results)
sum = 0
ans = 0
for (epsilon, accuracy) in results:
    if epsilon == 0:
        pass
    else:
        #print(epsilon, accuracy)
        sum += epsilon * accuracy
        ans += epsilon * accuracies[0]
print(sum)
print(ans)
print("Your Benchmark result is", sum/ans*100)