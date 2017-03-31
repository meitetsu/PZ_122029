import nltk

nltkDownloader = Downloader()
nltkDownloader.download()

# Plan:
#   1. Co pełni rolę x, y w przypadku tekstu?
#   2.
#   3. Zbadać zależność wyników dla SVM linear od liczby cech (features)
#   4. Zbadać zależność wyników dla SVM RBF od liczby cech(features)
#   5. Co moglibyśmy zrobić żeby zmniejszyć complexity?

# Plan obliczeń dla korpusu REUTERS
#   1. Co pełni rolę x, y w przypadku tekstu?
#   2. Zbadać zależność wyników dla SVM linear od liczby cech (features)
#   3. Złożoność obliczeniowa klasyfikacji?

# Program się zawiesza jeśli feature = każdy unikalny wyraz (term). Moglibyśmy wrzucić stopwatch.
#
# Wszystkie features w naszych warunkach niereailstycznych.
#   4. Zbadać zależność wyników dla SVM RBF od liczby cech (features) - jak je wybrać jeśli chcemy zatrzymać 300 features?
#   5. Co moglibyśmy zrobić żeby zmniejszyć complexity? - wyrzucamy stopwords i lematyzujemy tekst (użycie stemera).
#   6. Czy RBF poprawia wyniki?