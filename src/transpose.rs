// Stolen from: https://stackoverflow.com/a/75477884

pub struct TransposeIter<I, T>
where
    I: IntoIterator<Item = T>,
{
    iterators: Vec<I::IntoIter>,
}

pub trait TransposableIter<I, T>
where
    Self: Sized,
    Self: IntoIterator<Item = I>,
    I: IntoIterator<Item = T>,
{
    fn transpose(self) -> TransposeIter<I, T> {
        let iterators: Vec<_> = self.into_iter().map(|i| i.into_iter()).collect();
        TransposeIter { iterators }
    }
}

impl<I, T> Iterator for TransposeIter<I, T>
where
    I: IntoIterator<Item = T>,
{
    type Item = Vec<T>;
    fn next(&mut self) -> Option<Self::Item> {
        let output: Option<Vec<T>> = self.iterators.iter_mut().map(|iter| iter.next()).collect();
        output
    }
}

impl<I, T, Any> TransposableIter<I, T> for Any
where
    Any: IntoIterator<Item = I>,
    I: IntoIterator<Item = T>,
{
}
