#ifndef CIRCULAR_VECTOR_H
#define CIRCULAR_VECTOR_H

#include <cmath>
#include <deque>
#include <optional>
#include <stdexcept>
#include <vector>

template <class T>
class circular_vector {
  std::vector<T> vec{};
  // The first index for the circular vector in vec
  uint64_t first_index_vec = 0;
  // The first index overall num, like if this had been a regular expanding vector, what would the first item's index be
  uint64_t first_index_num = 0;
  // The last index for the circular vector in vec
  std::optional<uint64_t> last_index_vec = std::nullopt;

  uint64_t reclaimable_slots = 0;

  void realloc_vec(uint64_t new_capacity) {
    const auto used_size = size();
    if (new_capacity < used_size) return;
    std::vector<T> new_vec{};

    if (used_size > 0) {
      new_vec.reserve(new_capacity);
      new_vec.resize(used_size);
      auto begin_iter = begin();
      auto end_iter = end();
      std::move(begin_iter, end_iter, new_vec.begin());
    }

    vec = std::move(new_vec);
    // Reset everything so the new vec is now accessed in a non-circular way, at least until needed again
    last_index_vec = std::nullopt;
    first_index_vec = 0;
    reclaimable_slots = 0;
  }

public:
  using size_type = typename std::vector<T>::size_type;

  explicit circular_vector() = default;

  template <class U>
  class const_iterator {
  public:
    using difference_type = std::ptrdiff_t;
    using value_type = U;

  private:
    const std::vector<value_type>* vec = nullptr;
    uint64_t index = 0;  // vec->size() on the index means end/cend
    const uint64_t* first_index = nullptr;
    const std::optional<uint64_t>* last_index = nullptr;

    uint64_t add_to_index(difference_type add) const {
      const auto added_index = index + add;
      const auto vec_size = vec->size();
      const bool has_last_index = last_index != nullptr && last_index->has_value();
      // We are already on an index that has wrapped around the circular vector
      if (has_last_index&& index < *first_index) {
        // Given that we are wrapping around already, we don't allow wrapping around again,
        // if we reach the end or exceed it, just point to the end
        return added_index > **last_index ? vec_size : added_index;
      }
      // We have yet to wrap around (if that's even possible)
      else {
        // Wrapping around is not even needed
        if (added_index < vec_size) {
          // Return the added index, unless we have a last_index not before the first_index, and we would be going past that last_index,
          // in which case we return the vec_size which will result in an end iterator
          return (has_last_index && **last_index >= *first_index && added_index > **last_index) ? vec_size : added_index;
        }

        if (
          // If the vector doesn't have circular wrapping around behavior then just return index for end of vec
          !has_last_index ||
          // If the last index we have is larger than the first_index then wrapping around is not even possible
          **last_index >= *first_index ||
          // If we were to do a whole loop, just return index for end/cend, as we need to stop at some point.
          add >= vec_size
          ) {
          return vec_size;
        }

        const auto wrapped_index = added_index % vec_size;
        // If even wrapping around we went too far we stop at the last_index + 1 as well
        return wrapped_index > **last_index ? vec_size : wrapped_index;
      }
    }

    uint64_t remove_from_index(difference_type substract) const {
      if (substract == 0) return index;

      const auto vec_size = vec->size();
      const bool has_last_index = last_index != nullptr && last_index->has_value();
      const auto subtracted_index = static_cast<difference_type>(has_last_index && index == vec_size ? **last_index + 1 : index) + substract;
      // We are already on an index that has wrapped around the circular vector
      if (has_last_index && index < *first_index) {
        // If it's not enough to wrap around in reverse, then just return the new index
        if (subtracted_index >= 0) return subtracted_index;
        // Wrap around in reverse, remember that subtracted_index is negative here
        const auto wrapped_index = vec_size + subtracted_index;
        // If wrapping in reverse gets us before the *first_index, we reversed back to the beginning, stop there
        return wrapped_index >= *first_index ? wrapped_index : *first_index;
      }
      else if (index == vec_size) {
        return has_last_index ? **last_index : vec_size - 1;
      }
      // We have yet to wrap around (if that's even possible)
      else {
        // As we can't wrap around in reverse, we simply reverse as far as the first_index if set, 0 otherwise
        if (first_index != nullptr) {
          return std::max<difference_type>(subtracted_index, *first_index);
        }
        else {
          return std::max<difference_type>(subtracted_index, 0);
        }
      }
    }

    uint64_t shift_index(difference_type diff) const {
      return diff >= 0 ? add_to_index(diff) : remove_from_index(diff);
    }

    bool index_pos_larger_than(difference_type other_index) const {
      // Checks if this iterator's index position is larger than a given index, accounting for circular behavior
      // Prerequisite: this->index != other_index

      // If we enter here then it means this iterator's index is NOT from an element that has wrapped around the circular_vector
      if (first_index == nullptr || index >= *first_index) {
        const bool has_last_index = last_index != nullptr && !last_index->has_value();
        // If the vector is not circular yet, the index must be numerically larger than the other one and that's it
        if (!has_last_index && index > other_index) return true;

        // If there is circular behavior, then other_index might be numerically smaller but refer to a later element.
        // If the other_index is larger than the first_index but still smaller than this->index, then it's for a prior position,
        // if it's larger than first_index and also larger than this->index then the other one is for a later position,
        // otherwise despite other_index being numerically smaller its actually for a later position on the circular_index
        // as it refers to an element after wrapping around circularly
        if (other_index >= *first_index && index > other_index) return true;
        return false;
      }
      // conversely, if we are here it's that this iterator's index is from an element that IS after wrapping around the circular_vector

      // Now it's quite simple, if the other_index is from before wrapping around then it must be from a prior position,
      // else then we just need to compare them numerically.
      if (other_index >= *first_index) return true;
      return index > other_index;
    }

  public:
    const_iterator(const std::vector<value_type>* _vec, uint64_t _index, const uint64_t* _first_index, const std::optional<uint64_t>* _last_index)
      : vec(_vec), index(_index), first_index(_first_index), last_index(_last_index) {
      if ((first_index != nullptr || last_index != nullptr) && (first_index == nullptr || last_index == nullptr)) {
        throw std::runtime_error("circular_vector::iterator: first_index and last_index need to be both set or unset, no mixing");
      }
    }
    const_iterator() = default;

    uint64_t get_index() const { return index; }

    // Forward iterator requirements
    const value_type& operator*() const { return (*vec)[index]; }
    bool operator==(const const_iterator& other) const { return other.index == this->index && other.vec == this->vec; }

    const_iterator& operator++() {
      index = shift_index(1);
      return *this;
    }
    const_iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    // Bidirectional iterator requirements
    const_iterator& operator--() {
      index = shift_index(-1);
      return *this;
    }
    const_iterator operator--(int) {
      auto tmp = *this;
      --*this;
      return tmp;
    }

    // Random access iterator requirements
    const value_type& operator[](difference_type rhs) const { return (*vec)[shift_index(rhs)]; }

    const_iterator operator+(difference_type rhs) const { return const_iterator(this->vec, shift_index(rhs), first_index, last_index); }
    friend const_iterator operator+(difference_type lhs, const const_iterator& rhs) { return const_iterator(rhs.vec, rhs.shift_index(lhs), rhs.first_index, rhs.last_index); }
    const_iterator& operator+=(difference_type rhs) { index = shift_index(rhs); return *this; }

    difference_type operator-(const const_iterator& rhs) const { return shift_index(-rhs.index); }
    const_iterator operator-(difference_type rhs) const { return const_iterator(this->vec, shift_index(-rhs), first_index, last_index); }
    friend const_iterator operator-(difference_type lhs, const const_iterator& rhs) { return const_iterator(rhs.vec, rhs.shift_index(-lhs), rhs.first_index, rhs.last_index); }
    const_iterator& operator-=(difference_type rhs) { index = shift_index(-rhs); return *this; }

    bool operator>(const const_iterator& rhs) const { return this->index != rhs.index && index_pos_larger_than(rhs.index); }
    bool operator>=(const const_iterator& rhs) const { return this->index == rhs.index || index_pos_larger_than(rhs.index); }
    bool operator<(const const_iterator& rhs) const { return this->index != rhs.index && !index_pos_larger_than(rhs.index); }
    bool operator<=(const const_iterator& rhs) const { return this->index == rhs.index || !index_pos_larger_than(rhs.index); }
  };
  static_assert(std::random_access_iterator<const_iterator<uint64_t>>);

  size_type get_last_index_vec() const { return last_index_vec.has_value() ? *last_index_vec : vec.size() - 1; }
  size_type get_last_index_num() const { return first_index_num + this->size() - 1; }
  uint64_t get_index(const const_iterator<T>& iter) {
    const auto iter_index = iter.get_index();
    if (iter_index >= first_index_vec) {
      const auto diff = iter_index - first_index_vec;
      return first_index_num + diff;
    }
    const auto non_wrapped_element_count = vec.size() - first_index_vec;
    return first_index_num + non_wrapped_element_count + iter_index;
  }

  T& operator[](size_type pos) {
    // Check we are not trying to access out of bounds
    const auto last_allowed_pos = get_last_index_num();
    if (pos < first_index_num || pos > last_allowed_pos) {
      throw std::runtime_error("Can't access out of bounds index on circular_vector");
    }

    // Finally, get the index, wrapping around if necessary, and return the element reference
    const auto relative_pos = pos - first_index_num;
    const auto in_vec_index = (first_index_vec + relative_pos) % vec.size();
    return vec[in_vec_index];
  }

  const_iterator<T> begin() const { return const_iterator<T>(&vec, first_index_vec, &first_index_vec, &last_index_vec); }
  const_iterator<T> end() const { return const_iterator<T>(&vec, vec.size(), &first_index_vec, &last_index_vec); }
  const_iterator<T> cbegin() const { return const_iterator<T>(&vec, first_index_vec, &first_index_vec, &last_index_vec); }
  const_iterator<T> cend() const { return const_iterator<T>(&vec, vec.size(), &first_index_vec, &last_index_vec); }

  // The size including reclaimed/removed items, as it would have been in a regular vector
  size_type fullSize() const {
    return first_index_num + size();
  }
  size_type innerVecSize() const { return vec.size(); }
  size_type size() const {
    if (!last_index_vec.has_value() || *last_index_vec < first_index_vec) {
      return vec.size() - reclaimable_slots;
    }
    else {
      return *last_index_vec - first_index_vec + 1;
    }
  }
  bool empty() const { return size() == 0; }
  void clear() {
    first_index_vec = 0;
    last_index_vec = std::nullopt;
    reclaimable_slots = 0;
    vec = std::vector<T>();
  }

  void pop_front() {
    if (last_index_vec.has_value() && *last_index_vec == first_index_vec) {
      // Popped the last element! reset all circular behavior stuff and quit
      clear();
      return;
    }
    first_index_vec++;
    if (first_index_vec == vec.size()) {
      first_index_vec = 0;
    }
    first_index_num++;
    reclaimable_slots++;
  }
  void pop_back() {
    if (last_index_vec.has_value()) {
      if (*last_index_vec == first_index_vec) {
        // Popped the last element! reset all circular behavior stuff and quit
        clear();
        return;
      }
      if (*last_index_vec == 0) {
        last_index_vec = std::nullopt;
      }
      else {
        (*last_index_vec)--;
      }
      reclaimable_slots++;
    }
    else {
      vec.pop_back();
    }
  }
  void emplace_back(T&& chunk) {
    // If we can still expand without realloc just do it
    if (vec.empty() || vec.size() < vec.capacity()) {
      vec.emplace_back(std::move(chunk));
    }
    // If vec is full but some slots are reclaimable, we do circular buffer style usage
    else if (reclaimable_slots > 0) {
      if (last_index_vec.has_value()) {
        (*last_index_vec)++;
        if (*last_index_vec == vec.size()) {
          *last_index_vec = 0;
        }
      }
      else {
        last_index_vec = 0;
      }
      vec[*last_index_vec] = std::move(chunk);
      reclaimable_slots--;
    }
    // max capacity and no reclaimable_slots, realloc unavoidable
    else {
      realloc_vec(static_cast<uint64_t>(std::ceil(static_cast<double>(vec.capacity()) * 1.5)));
      vec.emplace_back(std::move(chunk));
    }
  }
  void emplace_back(T& chunk) {
    T chunk_copy = chunk;
    emplace_back(std::move(chunk_copy));
  }

  void shrink_to_fit() {
    // we check that shrinking is worth it, at the very least we check that we shouldn't need to realloc again
    // if a few elements are added
    const auto current_capacity = vec.capacity();
    const auto target_capacity = static_cast<uint64_t>(std::ceil(static_cast<double>(current_capacity) / 1.5));
    if (target_capacity > size()) {
      realloc_vec(target_capacity);
    }
  }

  T& front() { return vec[first_index_vec]; }
  const T& front() const { return vec[first_index_vec]; }
  T& back() { return vec[get_last_index_vec()]; }
  const T& back() const { return vec[get_last_index_vec()]; }
};
static_assert(std::ranges::range<circular_vector<uint64_t>>);

template <class T>
class circular_vector_debug {
  std::deque<T> instructions_deque;
  circular_vector<T*> instructions_vec;

  void check_instructions_equal(const T& instruction1, const T& instruction2) const {
    if (instruction1 != instruction2) {
      throw std::runtime_error("Verification error");
    }
  }

  void check_iterators_equal(auto& deque_iter1, auto& vec_iter2) const {
    auto deque_begin = instructions_deque.begin();
    auto deque_end = instructions_deque.end();
    auto deque_cbegin = instructions_deque.cbegin();
    auto deque_cend = instructions_deque.cend();
    auto vec_begin = instructions_vec.begin();
    auto vec_end = instructions_vec.end();
    auto vec_cbegin = instructions_vec.cbegin();
    auto vec_cend = instructions_vec.cend();

    const bool deque_iter1_is_begin = deque_iter1 == deque_begin;
    const bool deque_iter1_is_cbegin = deque_iter1 == deque_cbegin;
    const bool deque_iter1_is_end = deque_iter1 == deque_end;
    const bool deque_iter1_is_cend = deque_iter1 == deque_cend;
    const bool vec_iter1_is_begin = vec_iter2 == vec_begin;
    const bool vec_iter1_is_cbegin = vec_iter2 == vec_cbegin;
    const bool vec_iter1_is_end = vec_iter2 == vec_end;
    const bool vec_iter1_is_cend = vec_iter2 == vec_cend;
    if (
      deque_iter1_is_begin && !vec_iter1_is_begin ||
      !deque_iter1_is_begin && vec_iter1_is_begin ||
      deque_iter1_is_cbegin && !vec_iter1_is_cbegin ||
      !deque_iter1_is_cbegin && vec_iter1_is_cbegin ||
      deque_iter1_is_end && !vec_iter1_is_end ||
      !deque_iter1_is_end && vec_iter1_is_end ||
      deque_iter1_is_cend && !vec_iter1_is_cend ||
      !deque_iter1_is_cend && vec_iter1_is_cend
      ) {
      throw std::runtime_error("Verification error");
    }
    if (deque_iter1 == deque_end || deque_iter1 == deque_cend) return;
    auto& instruction1 = *deque_iter1;
    auto& instruction2 = **vec_iter2;
    check_instructions_equal(instruction1, instruction2);
  }

  void paranoid_check() const {
    auto deque_size = instructions_deque.size();
    auto vec_size = instructions_vec.size();
    if (deque_size != vec_size) {
      throw std::runtime_error("Verification error");
    }

    if (deque_size == 0) return;

    {
      auto& instruction1 = instructions_deque.front();
      auto& instruction2 = *instructions_vec.front();
      check_instructions_equal(instruction1, instruction2);
    }
    {
      auto& instruction1 = instructions_deque.back();
      auto& instruction2 = *instructions_vec.back();
      check_instructions_equal(instruction1, instruction2);
    }
    {
      auto deque_iter_begin = instructions_deque.begin();
      auto vec_iter_begin = instructions_vec.begin();
      check_iterators_equal(deque_iter_begin, vec_iter_begin);
      check_instructions_equal(**vec_iter_begin, *instructions_vec.front());
    }
    {
      auto deque_iter_cbegin = instructions_deque.cbegin();
      auto vec_iter_cbegin = instructions_vec.cbegin();
      check_iterators_equal(deque_iter_cbegin, vec_iter_cbegin);
      check_instructions_equal(**vec_iter_cbegin, *instructions_vec.front());
    }
    {
      auto deque_iter_end = instructions_deque.end();
      auto vec_iter_end = instructions_vec.end();
      check_iterators_equal(deque_iter_end, vec_iter_end);
      check_instructions_equal(**(vec_iter_end - 1), *instructions_vec.back());
    }
    {
      auto deque_iter_cend = instructions_deque.cend();
      auto vec_iter_cend = instructions_vec.cend();
      check_iterators_equal(deque_iter_cend, vec_iter_cend);
      check_instructions_equal(**(vec_iter_cend - 1), *instructions_vec.back());
    }
  }

public:
  using size_type = typename std::vector<T>::size_type;

  explicit circular_vector_debug() = default;

  T& operator[](size_type pos) {
    paranoid_check();
    auto& instruction1 = instructions_deque[pos];
    auto& instruction2 = *instructions_vec[pos];
    check_instructions_equal(instruction1, instruction2);
    paranoid_check();
    return instruction1;
  }

  typename std::deque<T>::iterator begin() {
    paranoid_check();
    typename std::deque<T>::iterator deque_iter = instructions_deque.begin();
    auto vec_iter = instructions_vec.begin();
    check_iterators_equal(deque_iter, vec_iter);
    paranoid_check();
    return deque_iter;
  }
  typename std::deque<T>::iterator end() {
    paranoid_check();
    typename std::deque<T>::iterator deque_iter = instructions_deque.end();
    auto vec_iter = instructions_vec.end();
    check_iterators_equal(deque_iter, vec_iter);
    paranoid_check();
    return deque_iter;
  }
  typename std::deque<T>::const_iterator cbegin() const {
    paranoid_check();
    typename std::deque<T>::const_iterator deque_iter = instructions_deque.cbegin();
    auto vec_iter = instructions_vec.cbegin();
    check_iterators_equal(deque_iter, vec_iter);
    paranoid_check();
    return deque_iter;
  }
  typename std::deque<T>::const_iterator cend() const {
    paranoid_check();
    typename std::deque<T>::const_iterator deque_iter = instructions_deque.cend();
    auto vec_iter = instructions_vec.cend();
    check_iterators_equal(deque_iter, vec_iter);
    paranoid_check();
    return deque_iter;
  }

  size_type size() const {
    paranoid_check();
    auto deque_size = instructions_deque.size();
    auto vec_size = instructions_vec.size();
    if (deque_size != vec_size) {
      throw std::runtime_error("Verification error");
    }
    paranoid_check();
    return deque_size;
  }
  void pop_front() {
    paranoid_check();
    size();
    auto& instruction1 = instructions_deque.front();
    auto& instruction2 = *instructions_vec.front();
    check_instructions_equal(instruction1, instruction2);
    instructions_deque.pop_front();
    instructions_vec.pop_front();
    size();
    paranoid_check();
  }
  void pop_back() {
    paranoid_check();
    size();
    auto& instruction1 = instructions_deque.back();
    auto& instruction2 = *instructions_vec.back();
    check_instructions_equal(instruction1, instruction2);
    instructions_deque.pop_back();
    instructions_vec.pop_back();
    size();
    paranoid_check();
  }
  void emplace_back(T&& instruction) {
    paranoid_check();
    size();
    instructions_deque.emplace_back(std::move(instruction));
    instructions_vec.emplace_back(&instructions_deque.back());
    size();

    back();
    paranoid_check();
  }
  void emplace_back(T& instruction) {
    T copy = instruction;
    emplace_back(std::move(copy));
  }

  void shrink_to_fit() {
    paranoid_check();
    size();
    instructions_deque.shrink_to_fit();
    instructions_vec.shrink_to_fit();
    size();
    paranoid_check();
  }

  T& front() {
    paranoid_check();
    auto& instruction1 = instructions_deque.front();
    auto& instruction2 = *instructions_vec.front();
    check_instructions_equal(instruction1, instruction2);
    paranoid_check();
    return instruction1;
  }
  const T& front() const {
    paranoid_check();
    auto& instruction1 = instructions_deque.front();
    auto& instruction2 = *instructions_vec.front();
    check_instructions_equal(instruction1, instruction2);
    paranoid_check();
    return instruction1;
  }
  T& back() {
    paranoid_check();
    auto& instruction1 = instructions_deque.back();
    auto& instruction2 = instructions_vec.back();
    check_instructions_equal(instruction1, *instruction2);
    paranoid_check();
    return instruction1;
  }
  const T& back() const {
    paranoid_check();
    auto& instruction1 = instructions_deque.back();
    auto& instruction2 = instructions_vec.back();
    check_instructions_equal(instruction1, *instruction2);
    paranoid_check();
    return instruction1;
  }
};

#endif
