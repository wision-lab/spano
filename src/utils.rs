#[allow(dead_code)]
pub fn product<'a: 'c, 'b: 'c, 'c, T>(
    xs: &'a [T],
    ys: &'b [T],
) -> impl Iterator<Item = (&'a T, &'b T)> + 'c {
    xs.iter().flat_map(move |x| std::iter::repeat(x).zip(ys))
}
