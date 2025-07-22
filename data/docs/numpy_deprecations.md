# NumPy deprecations reference guide

## Version 1.4.0

**correlate**
- **Replacement**: Use `old_behavior=False` parameter or migrate to new correlation definition
- **Context**: Takes a new keyword argument `old_behavior`. When `True` (default), returns same result as before. When `False`, computes conventional correlation and takes conjugate for complex arrays. Old behavior removed in NumPy 1.5.

**unique1d**
- **Replacement**: `numpy.unique`
- **Context**: Deprecated to consolidate unique functionality into a single function. Raises deprecation warning in 1.4, removed in 1.5.

**intersect1d_nu**
- **Replacement**: `numpy.intersect1d`
- **Context**: The "_nu" suffix was for "not unique" but functionality was merged into main function. Raises deprecation warning in 1.4, removed in 1.5.

**setmember1d**
- **Replacement**: `numpy.in1d`
- **Context**: Renamed to reflect that it's no longer a pure set operation and accepts arrays with duplicates. Raises deprecation warning in 1.4, removed in 1.5.

## Version 1.5.0

**No major deprecations were introduced in NumPy 1.5.0**

## Version 1.6.0

**numpy.fft.refft, refft2, refftn, irefft, irefft2, irefftn**
- **Replacement**: Functions without the 'e' prefix (e.g., `numpy.fft.rfft` instead of `numpy.fft.refft`)
- **Context**: These were aliases for the same functions without the 'e' in the name. Removed to reduce API confusion.

**numpy.memmap.sync(), numpy.memmap.close()**
- **Replacement**: Use `flush()` and `del memmap` instead
- **Context**: Methods removed from memmap objects. Use `flush()` for syncing and explicit deletion for closing.

**numpy.lib.ufunclike.log2**
- **Replacement**: Use `numpy.log2`
- **Context**: Moved from lib.ufunclike to main namespace for consistency.

**numpy.get_numpy_include**
- **Replacement**: `numpy.get_include`
- **Context**: Function name standardized to match common naming conventions.

**histogram "normed" keyword**
- **Replacement**: Use "density" keyword
- **Context**: The "normed" keyword was confusing and replaced with more descriptive "density" parameter.

## Version 1.7.0

**Custom string formatter with _format array attribute**
- **Replacement**: Use `formatter` parameter in `numpy.set_printoptions` or `numpy.array2string`
- **Context**: Custom string formatting method replaced with more flexible formatter system.

**concatenate axis behavior for 1D arrays**
- **Replacement**: Explicitly specify `axis=0` for 1D arrays
- **Context**: Previously ignored axis argument for 1D arrays. Now raises DeprecationWarning when axis != 0.

**Direct access to PyArrayObject* fields**
- **Replacement**: Use official C-API accessor functions
- **Context**: Direct field access deprecated in preparation for NumPy 2.0. Use PyArray_* accessor functions instead.

**Macros in old_defines.h**
- **Replacement**: Use newer macro versions (see tools/replace_old_macros.sed)
- **Context**: Old C-API macros deprecated. Will be removed in NumPy 2.0.

**NPY_CHAR member of NPY_TYPES enum**
- **Replacement**: Use appropriate string dtype instead
- **Context**: NPY_CHAR enum member causes confusion and will be removed in NumPy 1.8.

## Version 1.8.0

**numpy.linalg.qr 'full' and 'economic' modes**
- **Replacement**: Use 'complete' and 'reduced' modes respectively
- **Context**: Mode names changed for clarity. 'reduced' mode replaces 'full' and is the default. 'complete' mode replaces 'economic'.

**Non-integer indices and arguments**
- **Replacement**: Use proper integer values
- **Context**: Float indices and function arguments were previously truncated to integers silently. Now raises DeprecationWarning, will raise error in future versions.

**Boolean indices behavior**
- **Replacement**: Use proper integer indices or mask arrays
- **Context**: Boolean indices were treated as integers (0 or 1). Will be treated as masks in future versions.

**AdvancedNew iterator oa_ndim usage**
- **Replacement**: Use `oa_ndim = -1` to indicate no op_axes and itershape
- **Context**: The `oa_ndim == 0` case now indicates 0-D iteration. Old usage deprecated.

## Version 1.9.0

**Non-integer scalars for sequence repetition**
- **Replacement**: Use integer values for repetition
- **Context**: Using non-integer numpy scalars like `np.float_(2) * [1]` deprecated. Will raise error in future versions.

**select function integer and empty input**
- **Replacement**: Use boolean arrays for conditions
- **Context**: Integer and empty input to select deprecated. Only boolean arrays will be valid conditions in future.

**rank function**
- **Replacement**: Use `numpy.ndim` for array dimensions or `numpy.linalg.matrix_rank` for matrix rank
- **Context**: Deprecated to avoid confusion with `numpy.linalg.matrix_rank`.

**Object array equality comparisons**
- **Replacement**: Use explicit comparisons without relying on identity checks
- **Context**: Object array comparisons will not use identity checks in future. Use `arr is None` instead of `arr == None`.

**C-API functions npy_PyFile_Dup and npy_PyFile_DupClose**
- **Replacement**: Use `npy_PyFile_Dup2` and `npy_PyFile_DupClose2`
- **Context**: Original functions broken by Python 3 internal buffering. New versions declared in npy_3kcompat.h.

## Version 1.10.0

**Boolean array indexing edge cases**
- **Replacement**: Use proper boolean arrays matching array dimensions
- **Context**: Some edge cases with boolean indexing were fixed

**np.split behavior with empty arrays**
- **Replacement**: Prepare for dimension preservation in splits
- **Context**: Empty arrays in results will preserve original dimensions

**Scalar boolean indexing**
- **Replacement**: Use `array(True)` and `array(False)` as boolean indices only
- **Context**: Previously could be used equivalent to 1 and 0, now raises error

**Non-integer array-like indexing**
- **Replacement**: Cast object arrays of custom integer-like objects explicitly
- **Context**: All non-integer array-likes deprecated for indexing

**Multiple ellipsis in indexing**
- **Replacement**: Use only one ellipsis (...) per indexing operation
- **Context**: Multiple ellipses in indexing deprecated

**Non-integer axis indexes for reduction ufuncs**
- **Replacement**: Use integer axis values for reductions like `add.reduce` or `sum`
- **Context**: Non-integer axis indexes deprecated for safety

## Version 1.11.0

**Boolean indexing with non-boolean arrays**
- **Replacement**: Use proper boolean arrays for indexing
- **Context**: Prevented subtle bugs in indexing operations

**np.SafeEval class**
- **Replacement**: Use `ast.literal_eval` instead
- **Context**: SafeEval class removed in NumPy 1.11

**np.alterdot and np.restoredot functions**
- **Replacement**: These functions removed in NumPy 1.11
- **Context**: Functions for changing dot behavior were removed

**Default casting for inplace operations**
- **Replacement**: Prepare for 'same_kind' casting as default
- **Context**: Change from 'unsafe' to 'same_kind' casting for safety

## Version 1.12.0

**Assignment between structured arrays with different field names**
- **Replacement**: Prepare for position-based assignment in NumPy 1.14
- **Context**: Future change will assign fields by position rather than by name

**np.binary_repr with insufficient width parameter**
- **Replacement**: Provide sufficient width or remove width parameter
- **Context**: Previously silently ignored insufficient width, now considered unsafe

**np.linspace with non-integer num parameter**
- **Replacement**: Ensure num can be safely interpreted as integer
- **Context**: Raises DeprecationWarning when num cannot be safely interpreted

**Unsafe data attribute assignment**
- **Replacement**: Avoid assigning to 'data' attribute directly
- **Context**: Inherently unsafe operation that will be removed

**NAT comparison behavior**
- **Replacement**: Prepare for NAT comparisons to always be False except NAT != NAT
- **Context**: Behavior change scheduled for NumPy 1.13

## Version 1.13.0

**ndarray.__getslice__**
- **Replacement**: Use `.__getitem__(slice(start, end))` instead
- **Context**: Mirrors numpy.ndarray behavior and accounts for nested arrays

**np.expand_dims with invalid axis**
- **Replacement**: Ensure axis satisfies `-a.ndim - 1 <= axis <= a.ndim`
- **Context**: Invalid axis values were incorrectly accepted

**np.ma.argsort with default axis on >2D arrays**
- **Replacement**: Use explicit `axis` argument
- **Context**: Default value inconsistent with rest of numpy

**np.ma.minimum.reduce and np.ma.maximum.reduce with default axis on >2D arrays**
- **Replacement**: Use explicit `axis` argument
- **Context**: Default values inconsistent with rest of numpy

**np.ma.MaskedArray.mini**
- **Replacement**: Use `np.ma.minimum.reduce` instead
- **Context**: Almost duplicates functionality of np.MaskedArray.min

**Single-argument form of np.ma.minimum and np.ma.maximum**
- **Replacement**: Use `np.ma.minimum.reduce(x)` instead of `np.ma.minimum(x)`
- **Context**: Consistency with how this would be done with np.minimum

**NPY_CHAR type number (C-API)**
- **Replacement**: Use updated type numbers
- **Context**: Deprecated since version 1.7, now raises runtime warnings

## Version 1.14.0

**np.bincount with minlength=None**
- **Replacement**: Use `minlength=0` instead
- **Context**: None was not a sensible default value for this parameter

**np.fromstring with default sep argument**
- **Replacement**: Use `np.frombuffer` directly for binary data
- **Context**: Default behavior silently encoded unicode strings as binary data

**style option of array2string in non-legacy printing mode**
- **Replacement**: Remove the style argument
- **Context**: Deprecated due to new array printing changes

**PyArray_SetUpdateIfCopyBase (C-API)**
- **Replacement**: Use `PyArray_SetWritebackIfCopyBase` for NumPy >= 1.14
- **Context**: UPDATEIFCOPY arrays are not compatible with PyPy

**UPDATEIFCOPY arrays**
- **Replacement**: Use WRITEBACKIFCOPY arrays for PyPy compatibility
- **Context**: UPDATEIFCOPY arrays are not compatible with PyPy

## Version 1.15.0

**np.loads (pickle function alias)**
- **Replacement**: Use `pickle.loads` directly
- **Context**: Aliases of builtin pickle functions deprecated in favor of unaliased pickle.func names

**np.core.numeric.load**
- **Replacement**: Use `pickle.load` directly
- **Context**: Aliases of builtin pickle functions deprecated in favor of unaliased pickle.func names

**np.core.numeric.loads**
- **Replacement**: Use `pickle.loads` directly
- **Context**: Aliases of builtin pickle functions deprecated in favor of unaliased pickle.func names

**np.ma.loads, np.ma.dumps**
- **Replacement**: Use `pickle.loads` and `pickle.dumps` directly
- **Context**: These functions already failed on python 3 when called with a string

**np.ma.load, np.ma.dump**
- **Replacement**: Use `pickle.load` and `pickle.dump` directly
- **Context**: These functions already failed on python 3 when called with a string

**Multidimensional indexing with non-tuple**
- **Replacement**: Use `arr[tuple(ind)]` instead of `arr[ind]` where `ind` is a list
- **Context**: Deprecated to avoid ambiguity in expressions like `arr[[[0, 1], [0, 1]]]`

**Imports from numpy.testing.utils**
- **Replacement**: Import from `numpy.testing` instead
- **Context**: The testing utilities module structure was reorganized

**Imports from numpy.testing.decorators**
- **Replacement**: Import from `numpy.testing` instead
- **Context**: The testing utilities module structure was reorganized

**numpy.nditer without context manager for writeable arrays**
- **Replacement**: Use `with np.nditer(...) as it:` or call `it.close()` explicitly
- **Context**: Required for proper writeback semantics management

**normed argument of np.histogram**
- **Replacement**: Use `density` parameter instead
- **Context**: Long deprecated since 1.6.0, now emits DeprecationWarning

## Version 1.16.0

**np.core.typeNA and np.core.sctypeNA**
- **Replacement**: `np.sctypeDict`
- **Context**: Type dictionaries were buggy and undocumented, scheduled for removal in 1.18

**np.asscalar**
- **Replacement**: `np.ndarray.item`
- **Context**: Function was an alias to the more powerful item method, not tested, and fails for scalars

**np.set_array_ops and np.get_array_ops**
- **Replacement**: `PyUFunc_ReplaceLoopBySignature`
- **Context**: Deprecated as part of NEP 15 along with C-API functions PyArray_SetNumericOps and PyArray_GetNumericOps

**np.unravel_index dims parameter**
- **Replacement**: Use `shape` keyword instead
- **Context**: The dims keyword argument was deprecated in favor of shape for consistency

**np.histogram normed argument**
- **Replacement**: Use `density` parameter
- **Context**: Previously deprecated but no warning was issued until 1.16.0

**Positive operator (+) on non-numerical arrays**
- **Replacement**: Explicit casting or conversion
- **Context**: Previously returned unconditional copy, now raises DeprecationWarning for non-numerical arrays

**Passing iterators to stack functions**
- **Replacement**: Convert iterator to list/array first
- **Context**: Iterators should be explicitly converted to sequences before stacking

## Version 1.17.0

**np.polynomial functions with float instead of int**
- **Replacement**: Use explicit integer values
- **Context**: Functions previously accepted float values if integral (1.0, 2.0), now deprecated for consistency

**np.distutils.exec_command and temp_file_name**
- **Replacement**: `subprocess.Popen` and `tempfile.mkstemp`
- **Context**: Internal use refactored, better alternatives available

**Writeable flag of C-API wrapped arrays**
- **Replacement**: Manual flag setting will be required
- **Context**: Dangerous to force writeable flag, future versions will not allow switching to True from Python

**np.nonzero on 0d arrays**
- **Replacement**: `nonzero(atleast_1d(arr))`
- **Context**: Behavior was surprising and almost always incorrect usage

**Writing to result of np.broadcast_arrays**
- **Replacement**: Manually set writeable flag to True
- **Context**: Results have internal overlap making them unsafe to write to

## Version 1.18.0

**np.fromfile and np.fromstring error handling**
- **Replacement**: Proper error handling for bad data
- **Context**: Previously returned partial/invalid data silently, now will throw errors

**Non-scalar arrays as fill values in ma.fill_value**
- **Replacement**: Use scalar fill values
- **Context**: Broadcasting logic for non-scalar fill values is fragile, especially when slicing

**PyArray_As1D and PyArray_As2D**
- **Replacement**: `PyArray_AsCArray`
- **Context**: C-API functions deprecated in favor of more general alternative

**np.alen**
- **Replacement**: `len()`
- **Context**: Function was redundant with built-in len function

**Financial functions (fv, ipmt, irr, mirr, nper, npv, pmt, ppmt, pv, rate)**
- **Replacement**: `numpy-financial` package
- **Context**: Removed in accordance with NEP-32, available as separate package

**axis argument to np.ma.mask_cols and np.ma.mask_row**
- **Replacement**: Remove axis argument (it was ignored)
- **Context**: Argument was always ignored, so removal has no functional impact

## Version 1.19.0

**Automatic dtype=object for ragged input**
- **Replacement**: Explicit `dtype=object`
- **Context**: Prevents unexpected behavior per NEP 34, users should be explicit about object arrays

**shape=0 in np.rec factory functions**
- **Replacement**: Use `None` or explicit array length
- **Context**: 0 was special-cased as alias for None, will be treated as normal array length

**PyArray_GetArrayParamsFromObject, PyUFunc_GenericFunction, PyUFunc_SetUsesArraysAsData**
- **Replacement**: Convert to array, use `PyObject_Call`
- **Context**: Probably unused C-API functions deprecated for simplification

**Converting generic types to dtypes**
- **Replacement**: Use specific dtype names
- **Context**: Super classes like np.integer converted to np.int_ when np.int8, np.int16 etc. also valid

**round for np.complexfloating scalars**
- **Replacement**: Use `np.round` instead of builtin `round`
- **Context**: Builtin round deprecated for complex scalars, np.round unaffected

**np.ndarray.tostring()**
- **Replacement**: `np.ndarray.tobytes()`
- **Context**: Aligns with builtin array.array methods, tobytes existed since 1.9

## Version 1.20.0

**Aliases of builtin types (np.int, np.float, np.complex, etc.)**
- **Replacement**: Use `int`, `float`, `complex` or explicit NumPy types like `np.int64`
- **Context**: Cause confusion for newcomers, existed mainly for historic reasons

**shape=None to functions with non-optional shape argument**
- **Replacement**: Use `shape=()` explicitly
- **Context**: Previously was alias for shape=(), now requires explicit empty tuple

**Indexing errors when result is empty**
- **Replacement**: Ensure valid indices even for empty results
- **Context**: Out-of-bounds indices will be checked even when result would be empty

**Inexact matches for mode and searchside**
- **Replacement**: Use full string values ("clip" not "clap", "right" not "random")
- **Context**: Case insensitive and inexact matches caused confusion

**np.dual module**
- **Replacement**: Import functions directly from NumPy or SciPy
- **Context**: Module was redundant with direct imports

**outer and ufunc.outer for matrix**
- **Replacement**: Convert matrix to array first
- **Context**: np.matrix use with outer calls will require manual conversion

**Numeric Style types (Bytes0, Str0, Uint32, Uint64, Datetime64)**
- **Replacement**: Use lowercase variants ("S", "U", "uint32", "uint64", "datetime64")
- **Context**: Consistency with general NumPy naming conventions

**ndincr method of ndindex**
- **Replacement**: `next(it)` instead of `it.ndincr()`
- **Context**: Documentation warned against usage since NumPy 1.8

**ArrayLike objects without __len__ and __getitem__**
- **Replacement**: Implement sequence protocol or use explicit conversion
- **Context**: Objects defining array protocols but not sequence protocol will behave differently

## Version 1.21.0

**The .dtype attribute must return a dtype**
- **Replacement**: Ensure .dtype returns proper dtype object
- **Context**: NumPy will stop recursively coercing .dtype results that aren't dtypes

**Inexact matches for np.convolve and np.correlate**
- **Replacement**: Use full strings ("same", "valid", "full") instead of ("s", "v", "f")
- **Context**: Case insensitive and inexact matches for mode argument deprecated

**np.typeDict**
- **Replacement**: `np.sctypeDict`
- **Context**: Has been deprecated alias for over 14 years, finally issuing warnings

**Exceptions during array-like creation**
- **Replacement**: Raise AttributeError instead of other exceptions
- **Context**: Exceptions other than AttributeError during __array__ access will issue warnings

**Four ndarray.ctypes methods**
- **Replacement**: Use properties instead of methods
- **Context**: `_ctypes.get_data` → `_ctypes.data`, `_ctypes.get_shape` → `_ctypes.shape`, `_ctypes.get_strides` → `_ctypes.strides`, `_ctypes.get_as_parameter` → `_ctypes._as_parameter_`

## Version 1.22.0

**delimitor parameter in numpy.ma.mrecords.fromtextfile()**
- **Replacement**: Use `delimiter` parameter instead
- **Context**: Misspelled keyword argument was corrected, `delimitor` now emits deprecation warning

**Boolean values for kth parameter in numpy.partition and numpy.argpartition**
- **Replacement**: Use integer values instead
- **Context**: Previously accepted boolean values which were converted to integers

**numpy.MachAr class and finfo.machar attribute**
- **Replacement**: Access properties directly from numpy.finfo attributes
- **Context**: Deprecated to simplify API and direct access to floating-point information

## Version 1.23.0

**Setting __array_finalize__ to None**
- **Replacement**: Must be a method, may call `super().__array_finalize__(obj)`
- **Context**: Improves consistency in array finalization behavior

**axis=32 (axis=np.MAXDIMS) usage**
- **Replacement**: Use `axis=None` instead
- **Context**: axis=32 previously had same meaning as axis=None in many cases

**PyDataMem_SetEventHook function**
- **Replacement**: Use Python's built-in `tracemalloc` for allocation tracking
- **Context**: Hook function for memory allocation tracking is no longer needed

**numpy.distutils module**
- **Replacement**: Use standard Python packaging tools
- **Context**: Deprecated due to Python's distutils deprecation, will be removed 2 years after Python 3.12 release

**numpy.loadtxt with integer dtype for floating point formatted values**
- **Replacement**: Ensure data format matches requested dtype
- **Context**: Now gives DeprecationWarning when integer dtype requested for floating point values

## Version 1.24.0

**numpy.fastCopyAndTranspose function**
- **Replacement**: Use `arr.T.copy()` instead
- **Context**: Direct method calls are more explicit and efficient

**Out-of-bound Python integer conversions**
- **Replacement**: Use explicit casting methods like `arr.astype()` or `arr.view()`
- **Context**: Conversions like `np.uint8(-1)` will now check bounds and warn

**numpy.msort function**
- **Replacement**: Use `np.sort(a, axis=0)` instead
- **Context**: Dedicated merge sort function was redundant

**Scalar type aliases with 0 bit size**
- **Replacement**: Use proper dtype specifications
- **Context**: Aliases like `np.str0`, `np.bytes0`, `np.void0`, `np.int0`, `np.uint0`, `np.bool8` are deprecated

## Version 1.25.0

**np.core.MachAr (private API)**
- **Replacement**: Use public APIs for floating-point information
- **Context**: Private API should not be used directly by users

**np.finfo(None)**
- **Replacement**: Use `np.finfo` with specific dtype argument
- **Context**: None argument was not meaningful

**np.round_ function**
- **Replacement**: Use `np.round` instead
- **Context**: Underscore version was redundant

**np.product function**
- **Replacement**: Use `np.prod` instead
- **Context**: Shorter, more standard name

**np.cumproduct function**
- **Replacement**: Use `np.cumprod` instead
- **Context**: Shorter, more standard name

**np.sometrue function**
- **Replacement**: Use `np.any` instead
- **Context**: More descriptive function name

**np.alltrue function**
- **Replacement**: Use `np.all` instead
- **Context**: More descriptive function name

**Size-1 arrays treated as scalars**
- **Replacement**: Use `arr.item()` or `arr[0]` to extract scalar values
- **Context**: Only ndim-0 arrays should be treated as scalars

**np.find_common_type function**
- **Replacement**: Use `np.result_type` or `np.promote_types` instead
- **Context**: Better type promotion functions available

## Version 1.26.0

**No major new deprecations were introduced in NumPy 1.26.0**

## Version 2.0.0

**np.geterrobj, np.seterrobj**
- **Replacement**: Use context manager `with np.errstate():`
- **Context**: These functions and the related ufunc keyword argument `extobj=` have been removed to improve error handling patterns

**np.cast**
- **Replacement**: `np.asarray(arg, dtype=dtype)`
- **Context**: The literal replacement for `np.cast[dtype](arg)` is the more explicit asarray syntax

**np.source**
- **Replacement**: `inspect.getsource`
- **Context**: Removed in favor of Python's standard library function

**np.lookfor**
- **Replacement**: None provided
- **Context**: Removed from public API

**numpy.who**
- **Replacement**: Use variable explorer in IDEs like Spyder or Jupyter Notebook
- **Context**: Functionality is better served by modern development environments

**np.float_**
- **Replacement**: `np.float64`
- **Context**: Alias removed for clarity

**np.complex_**
- **Replacement**: `np.complex128`
- **Context**: Alias removed for clarity

**np.longfloat**
- **Replacement**: `np.longdouble`
- **Context**: More descriptive naming

**np.singlecomplex**
- **Replacement**: `np.complex64`
- **Context**: More descriptive naming

**np.cfloat**
- **Replacement**: `np.complex128`
- **Context**: More descriptive naming

**np.string_**
- **Replacement**: `np.bytes_`
- **Context**: Better reflects the actual data type

**np.unicode_**
- **Replacement**: `np.str_`
- **Context**: Better reflects the actual data type

**np.Inf, np.Infinity, np.infty**
- **Replacement**: `np.inf`
- **Context**: Standardization on single spelling

**np.NaN**
- **Replacement**: `np.nan`
- **Context**: Standardization on lowercase

**np.asfarray**
- **Replacement**: `np.asarray` with proper dtype
- **Context**: More explicit approach preferred

**np.safe_eval**
- **Replacement**: `ast.literal_eval`
- **Context**: Security and standardization on Python's standard library

**np.trapz**
- **Replacement**: `np.trapezoid` or `scipy.integrate` function
- **Context**: More descriptive naming

**np.in1d**
- **Replacement**: `np.isin`
- **Context**: More intuitive naming

**np.row_stack**
- **Replacement**: `np.vstack`
- **Context**: Elimination of redundant aliases

**__array_wrap__ signature**
- **Replacement**: `__array_wrap__(self, arr, context=None, return_scalar=False)`
- **Context**: Support for implementations not accepting all three parameters is deprecated

**FFT functions with conflicting parameters**
- **Replacement**: Pass sequence `[0, …, k-1]` to `axes` parameter for an array of dimension k
- **Context**: Alignment with array API standard when `s` parameter is not None and `axes` parameter is None

**assert_array_equal, assert_array_almost_equal keyword arguments**
- **Replacement**: Pass first two arguments as positional arguments instead of using `x` and `y` keywords
- **Context**: API simplification

## Version 2.1.0

**fix_imports in numpy.save**
- **Replacement**: Remove the parameter (it's ignored anyway)
- **Context**: Since NumPy 1.17, `numpy.save` uses a pickle protocol that no longer supports Python 2, making this keyword obsolete

**Non-integer inputs to bincount**
- **Replacement**: Cast inputs to integers explicitly before calling
- **Context**: Such inputs are silently cast to integers with no warning about precision loss, which is problematic

## Version 2.2.0

**No accessible information available for NumPy 2.2.0 deprecations**

## Version 2.3.0

**numpy.typing.mypy_plugin**
- **Replacement**: Remove from mypy configuration plugins section
- **Context**: Deprecated in favor of platform-agnostic static type inference

**numpy.typing.NBitBase**
- **Replacement**: Use `typing.overload` for type parameter bounds
- **Context**: Changes in NumPy 2.2.0 made `float64` and `complex128` concrete subtypes, breaking the previous pattern

**np.tostring**
- **Replacement**: `tobytes`
- **Context**: Method naming consistency (deprecated since 1.19)

**concatenate casting behavior**
- **Replacement**: Use explicit `casting` parameter
- **Context**: `concatenate()` with `axis=None` now uses `same-kind` casting by default instead of `unsafe`

**datetime64/timedelta64 construction**
- **Replacement**: Use two-tuple `(unit, num)` or four-tuple `(unit, num, den, 1)`
- **Context**: No longer accepts event value in tuple construction (deprecated since 1.14)

**Complex scalar round() function**
- **Replacement**: Use `np.round` or `scalar.round` instead of Python's built-in `round`
- **Context**: Python built-in `round` errors for complex scalars (deprecated since 1.19)