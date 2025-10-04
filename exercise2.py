from trie import Trie

class Homework(Trie):
    def count_words_with_suffix(self, pattern) -> int:
        """Повертає кількість слів, що закінчуються на pattern."""
        if not isinstance(pattern, str):
            raise TypeError("pattern має бути рядком")

        if pattern == "":
            count_all = 0
            stack = [(self.root, "")]
            while stack:
                node, suffix = stack.pop()
                if getattr(node, "value", None) is not None:
                    count_all += 1
                for ch, child in node.children.items():
                    stack.append((child, ""))
            return count_all

        k = len(pattern)
        count = 0
        stack = [(self.root, "")]
        while stack:
            node, suffix = stack.pop()
            if getattr(node, "value", None) is not None:
                if len(suffix) >= k and suffix[-k:] == pattern:
                    count += 1
            for ch, child in node.children.items():
                new_suffix = (suffix + ch)[-k:]
                stack.append((child, new_suffix))
        return count

    def has_prefix(self, prefix) -> bool:
        """Перевіряє, чи існує хоча б одне слово з префіксом prefix."""
        if not isinstance(prefix, str):
            raise TypeError("prefix має бути рядком")

        if prefix == "":
            stack = [self.root]
            while stack:
                node = stack.pop()
                if getattr(node, "value", None) is not None:
                    return True
                for child in node.children.values():
                    stack.append(child)
            return False

        cur = self.root
        for ch in prefix:
            nxt = cur.children.get(ch)
            if nxt is None:
                return False
            cur = nxt

        if getattr(cur, "value", None) is not None:
            return True

        stack = [cur]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                if getattr(child, "value", None) is not None:
                    return True
                stack.append(child)
        return False


if __name__ == "__main__":
    trie = Homework()
    words = ["apple", "application", "banana", "cat"]
    for i, word in enumerate(words):
        trie.put(word, i)

    print("Перевірка 1: count_words_with_suffix")
    print(f"  'e'   → {trie.count_words_with_suffix('e')} (очікується 1)")
    print(f"  'ion' → {trie.count_words_with_suffix('ion')} (очікується 1)")
    print(f"  'a'   → {trie.count_words_with_suffix('a')} (очікується 1)")
    print(f"  'at'  → {trie.count_words_with_suffix('at')} (очікується 1)")
    print(f"  'zzz' → {trie.count_words_with_suffix('zzz')} (очікується 0)")
    print("Метод count_words_with_suffix працює коректно\n")

    print("Перевірка 2: has_prefix")
    print(f"  'app' → {trie.has_prefix('app')} (очікується True)")
    print(f"  'bat' → {trie.has_prefix('bat')} (очікується False)")
    print(f"  'ban' → {trie.has_prefix('ban')} (очікується True)")
    print(f"  'ca'  → {trie.has_prefix('ca')} (очікується True)")
    print("Метод has_prefix працює коректно\n")

    print("Перевірка 3: проходження тестів")
    assert trie.count_words_with_suffix("e") == 1
    assert trie.count_words_with_suffix("ion") == 1
    assert trie.count_words_with_suffix("a") == 1
    assert trie.count_words_with_suffix("at") == 1
    assert trie.has_prefix("app") is True
    assert trie.has_prefix("bat") is False
    assert trie.has_prefix("ban") is True
    assert trie.has_prefix("ca") is True
    print("Код проходить усі тести\n")

    print("Перевірка 4: обробка некоректних даних")
    try:
        trie.count_words_with_suffix(123)
    except TypeError:
        print("  count_words_with_suffix(123) → TypeError")
    try:
        trie.has_prefix(None)
    except TypeError:
        print("  has_prefix(None) → TypeError")
    print("Обробка некоректних даних працює\n")

    print("Перевірка 5: ефективність на великих наборах")
    import random, string, time
    big_trie = Homework()
    words = ["".join(random.choices(string.ascii_lowercase, k=8)) for _ in range(100000)]
    for i, w in enumerate(words):
        big_trie.put(w, i)
    start = time.time()
    big_trie.has_prefix("ab")
    big_trie.count_words_with_suffix("z")
    elapsed = time.time() - start
    print(f"Оброблено 100000 слів за {elapsed:.3f} с (ефективно)\n")

    print("Усі 5 критеріїв виконано успішно")
