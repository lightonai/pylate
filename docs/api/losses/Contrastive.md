# Contrastive

Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.



## Parameters

- **model** (*[models.ColBERT](../../models/ColBERT)*)

    ColBERT model.

- **score_metric** – defaults to `<function colbert_scores at 0x7fe76c02f240>`

    ColBERT scoring function. Defaults to colbert_scores.

- **size_average** (*bool*) – defaults to `True`

    Average by the size of the mini-batch.



## Examples

```python
>>> from pylate import models, losses

>>> model = models.ColBERT(
...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
... )

>>> loss = losses.Contrastive(model=model)

>>> anchor = model.tokenize([
...     "fruits are healthy.",
... ], is_query=True)

>>> positive = model.tokenize([
...     "fruits are good for health.",
... ], is_query=False)

>>> negative = model.tokenize([
...     "fruits are bad for health.",
... ], is_query=False)

>>> sentence_features = [anchor, positive, negative]

>>> loss = loss(sentence_features=sentence_features)
>>> assert isinstance(loss.item(), float)
```

## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **args**
    - **kwargs**

???- note "add_module"

    Add a child module to the current module.

    The module can be accessed as an attribute using the given name.  Args:     name (str): name of the child module. The child module can be         accessed from this module using the given name     module (Module): child module to be added to the module.

    **Parameters**

    - **name**     (*str*)
    - **module**     (*Optional[ForwardRef('Module')]*)

???- note "apply"

    Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

    Typical use includes initializing the parameters of a model (see also :ref:`nn-init-doc`).  Args:     fn (:class:`Module` -> None): function to be applied to each submodule  Returns:     Module: self  Example::      >>> @torch.no_grad()     >>> def init_weights(m):     >>>     print(m)     >>>     if type(m) == nn.Linear:     >>>         m.weight.fill_(1.0)     >>>         print(m.weight)     >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))     >>> net.apply(init_weights)     Linear(in_features=2, out_features=2, bias=True)     Parameter containing:     tensor([[1., 1.],             [1., 1.]], requires_grad=True)     Linear(in_features=2, out_features=2, bias=True)     Parameter containing:     tensor([[1., 1.],             [1., 1.]], requires_grad=True)     Sequential(       (0): Linear(in_features=2, out_features=2, bias=True)       (1): Linear(in_features=2, out_features=2, bias=True)     )

    **Parameters**

    - **fn**     (*Callable[[ForwardRef('Module')], NoneType]*)

???- note "bfloat16"

    Casts all floating point parameters and buffers to ``bfloat16`` datatype.

    .. note::     This method modifies the module in-place.  Returns:     Module: self


???- note "buffers"

    Return an iterator over module buffers.

    Args:     recurse (bool): if True, then yields buffers of this module         and all submodules. Otherwise, yields only buffers that         are direct members of this module.  Yields:     torch.Tensor: module buffer  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> for buf in model.buffers():     >>>     print(type(buf), buf.size())     <class 'torch.Tensor'> (20L,)     <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

    **Parameters**

    - **recurse**     (*bool*)     – defaults to `True`

???- note "children"

    Return an iterator over immediate children modules.

    Yields:     Module: a child module


???- note "compile"

    Compile this Module's forward using :func:`torch.compile`.

    This Module's `__call__` method is compiled and all arguments are passed as-is to :func:`torch.compile`.  See :func:`torch.compile` for details on the arguments for this function.

    **Parameters**

    - **args**
    - **kwargs**

???- note "cpu"

    Move all model parameters and buffers to the CPU.

    .. note::     This method modifies the module in-place.  Returns:     Module: self


???- note "cuda"

    Move all model parameters and buffers to the GPU.

    This also makes associated parameters and buffers different objects. So it should be called before constructing optimizer if the module will live on GPU while being optimized.  .. note::     This method modifies the module in-place.  Args:     device (int, optional): if specified, all parameters will be         copied to that device  Returns:     Module: self

    **Parameters**

    - **device**     (*Union[int, torch.device, NoneType]*)     – defaults to `None`

???- note "double"

    Casts all floating point parameters and buffers to ``double`` datatype.

    .. note::     This method modifies the module in-place.  Returns:     Module: self


???- note "eval"

    Set the module in evaluation mode.

    This has any effect only on certain modules. See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`, etc.  This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.  See :ref:`locally-disable-grad-doc` for a comparison between `.eval()` and several similar mechanisms that may be confused with it.  Returns:     Module: self


???- note "extra_repr"

    Set the extra representation of the module.

    To print customized extra information, you should re-implement this method in your own modules. Both single-line and multi-line strings are acceptable.


???- note "float"

    Casts all floating point parameters and buffers to ``float`` datatype.

    .. note::     This method modifies the module in-place.  Returns:     Module: self


???- note "forward"

    Compute the Constrastive loss.

    **Parameters**

    - **sentence_features**     (*Iterable[dict[str, torch.Tensor]]*)
    - **labels**     (*torch.Tensor | None*)     – defaults to `None`

???- note "get_buffer"

    Return the buffer given by ``target`` if it exists, otherwise throw an error.

    See the docstring for ``get_submodule`` for a more detailed explanation of this method's functionality as well as how to correctly specify ``target``.  Args:     target: The fully-qualified string name of the buffer         to look for. (See ``get_submodule`` for how to specify a         fully-qualified string.)  Returns:     torch.Tensor: The buffer referenced by ``target``  Raises:     AttributeError: If the target string references an invalid         path or resolves to something that is not a         buffer

    **Parameters**

    - **target**     (*str*)

???- note "get_extra_state"

    Return any extra state to include in the module's state_dict.

    Implement this and a corresponding :func:`set_extra_state` for your module if you need to store extra state. This function is called when building the module's `state_dict()`.  Note that extra state should be picklable to ensure working serialization of the state_dict. We only provide provide backwards compatibility guarantees for serializing Tensors; other objects may break backwards compatibility if their serialized pickled form changes.  Returns:     object: Any extra state to store in the module's state_dict


???- note "get_parameter"

    Return the parameter given by ``target`` if it exists, otherwise throw an error.

    See the docstring for ``get_submodule`` for a more detailed explanation of this method's functionality as well as how to correctly specify ``target``.  Args:     target: The fully-qualified string name of the Parameter         to look for. (See ``get_submodule`` for how to specify a         fully-qualified string.)  Returns:     torch.nn.Parameter: The Parameter referenced by ``target``  Raises:     AttributeError: If the target string references an invalid         path or resolves to something that is not an         ``nn.Parameter``

    **Parameters**

    - **target**     (*str*)

???- note "get_submodule"

    Return the submodule given by ``target`` if it exists, otherwise throw an error.

    For example, let's say you have an ``nn.Module`` ``A`` that looks like this:  .. code-block:: text      A(         (net_b): Module(             (net_c): Module(                 (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))             )             (linear): Linear(in_features=100, out_features=200, bias=True)         )     )  (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested submodule ``net_b``, which itself has two submodules ``net_c`` and ``linear``. ``net_c`` then has a submodule ``conv``.)  To check whether or not we have the ``linear`` submodule, we would call ``get_submodule("net_b.linear")``. To check whether we have the ``conv`` submodule, we would call ``get_submodule("net_b.net_c.conv")``.  The runtime of ``get_submodule`` is bounded by the degree of module nesting in ``target``. A query against ``named_modules`` achieves the same result, but it is O(N) in the number of transitive modules. So, for a simple check to see if some submodule exists, ``get_submodule`` should always be used.  Args:     target: The fully-qualified string name of the submodule         to look for. (See above example for how to specify a         fully-qualified string.)  Returns:     torch.nn.Module: The submodule referenced by ``target``  Raises:     AttributeError: If the target string references an invalid         path or resolves to something that is not an         ``nn.Module``

    **Parameters**

    - **target**     (*str*)

???- note "half"

    Casts all floating point parameters and buffers to ``half`` datatype.

    .. note::     This method modifies the module in-place.  Returns:     Module: self


???- note "ipu"

    Move all model parameters and buffers to the IPU.

    This also makes associated parameters and buffers different objects. So it should be called before constructing optimizer if the module will live on IPU while being optimized.  .. note::     This method modifies the module in-place.  Arguments:     device (int, optional): if specified, all parameters will be         copied to that device  Returns:     Module: self

    **Parameters**

    - **device**     (*Union[int, torch.device, NoneType]*)     – defaults to `None`

???- note "load_state_dict"

    Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

    If :attr:`strict` is ``True``, then the keys of :attr:`state_dict` must exactly match the keys returned by this module's :meth:`~torch.nn.Module.state_dict` function.  .. warning::     If :attr:`assign` is ``True`` the optimizer must be created after     the call to :attr:`load_state_dict` unless     :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.  Args:     state_dict (dict): a dict containing parameters and         persistent buffers.     strict (bool, optional): whether to strictly enforce that the keys         in :attr:`state_dict` match the keys returned by this module's         :meth:`~torch.nn.Module.state_dict` function. Default: ``True``     assign (bool, optional): When ``False``, the properties of the tensors         in the current module are preserved while when ``True``, the         properties of the Tensors in the state dict are preserved. The only         exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s         for which the value from the module is preserved.         Default: ``False``  Returns:     ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:         * **missing_keys** is a list of str containing the missing keys         * **unexpected_keys** is a list of str containing the unexpected keys  Note:     If a parameter or buffer is registered as ``None`` and its corresponding key     exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a     ``RuntimeError``.

    **Parameters**

    - **state_dict**     (*Mapping[str, Any]*)
    - **strict**     (*bool*)     – defaults to `True`
    - **assign**     (*bool*)     – defaults to `False`

???- note "modules"

    Return an iterator over all modules in the network.

    Yields:     Module: a module in the network  Note:     Duplicate modules are returned only once. In the following     example, ``l`` will be returned only once.  Example::      >>> l = nn.Linear(2, 2)     >>> net = nn.Sequential(l, l)     >>> for idx, m in enumerate(net.modules()):     ...     print(idx, '->', m)      0 -> Sequential(       (0): Linear(in_features=2, out_features=2, bias=True)       (1): Linear(in_features=2, out_features=2, bias=True)     )     1 -> Linear(in_features=2, out_features=2, bias=True)


???- note "named_buffers"

    Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

    Args:     prefix (str): prefix to prepend to all buffer names.     recurse (bool, optional): if True, then yields buffers of this module         and all submodules. Otherwise, yields only buffers that         are direct members of this module. Defaults to True.     remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.  Yields:     (str, torch.Tensor): Tuple containing the name and buffer  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> for name, buf in self.named_buffers():     >>>     if name in ['running_var']:     >>>         print(buf.size())

    **Parameters**

    - **prefix**     (*str*)     – defaults to ``
    - **recurse**     (*bool*)     – defaults to `True`
    - **remove_duplicate**     (*bool*)     – defaults to `True`

???- note "named_children"

    Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

    Yields:     (str, Module): Tuple containing a name and child module  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> for name, module in model.named_children():     >>>     if name in ['conv4', 'conv5']:     >>>         print(module)


???- note "named_modules"

    Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

    Args:     memo: a memo to store the set of modules already added to the result     prefix: a prefix that will be added to the name of the module     remove_duplicate: whether to remove the duplicated module instances in the result         or not  Yields:     (str, Module): Tuple of name and module  Note:     Duplicate modules are returned only once. In the following     example, ``l`` will be returned only once.  Example::      >>> l = nn.Linear(2, 2)     >>> net = nn.Sequential(l, l)     >>> for idx, m in enumerate(net.named_modules()):     ...     print(idx, '->', m)      0 -> ('', Sequential(       (0): Linear(in_features=2, out_features=2, bias=True)       (1): Linear(in_features=2, out_features=2, bias=True)     ))     1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

    **Parameters**

    - **memo**     (*Optional[Set[ForwardRef('Module')]]*)     – defaults to `None`
    - **prefix**     (*str*)     – defaults to ``
    - **remove_duplicate**     (*bool*)     – defaults to `True`

???- note "named_parameters"

    Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

    Args:     prefix (str): prefix to prepend to all parameter names.     recurse (bool): if True, then yields parameters of this module         and all submodules. Otherwise, yields only parameters that         are direct members of this module.     remove_duplicate (bool, optional): whether to remove the duplicated         parameters in the result. Defaults to True.  Yields:     (str, Parameter): Tuple containing the name and parameter  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> for name, param in self.named_parameters():     >>>     if name in ['bias']:     >>>         print(param.size())

    **Parameters**

    - **prefix**     (*str*)     – defaults to ``
    - **recurse**     (*bool*)     – defaults to `True`
    - **remove_duplicate**     (*bool*)     – defaults to `True`

???- note "parameters"

    Return an iterator over module parameters.

    This is typically passed to an optimizer.  Args:     recurse (bool): if True, then yields parameters of this module         and all submodules. Otherwise, yields only parameters that         are direct members of this module.  Yields:     Parameter: module parameter  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> for param in model.parameters():     >>>     print(type(param), param.size())     <class 'torch.Tensor'> (20L,)     <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

    **Parameters**

    - **recurse**     (*bool*)     – defaults to `True`

???- note "register_backward_hook"

    Register a backward hook on the module.

    This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and the behavior of this function will change in future versions.  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Callable[[ForwardRef('Module'), Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, Tuple[torch.Tensor, ...], torch.Tensor]]*)

???- note "register_buffer"

    Add a buffer to the module.

    This is typically used to register a buffer that should not to be considered a model parameter. For example, BatchNorm's ``running_mean`` is not a parameter, but is part of the module's state. Buffers, by default, are persistent and will be saved alongside parameters. This behavior can be changed by setting :attr:`persistent` to ``False``. The only difference between a persistent buffer and a non-persistent buffer is that the latter will not be a part of this module's :attr:`state_dict`.  Buffers can be accessed as attributes using given names.  Args:     name (str): name of the buffer. The buffer can be accessed         from this module using the given name     tensor (Tensor or None): buffer to be registered. If ``None``, then operations         that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,         the buffer is **not** included in the module's :attr:`state_dict`.     persistent (bool): whether the buffer is part of this module's         :attr:`state_dict`.  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> self.register_buffer('running_mean', torch.zeros(num_features))

    **Parameters**

    - **name**     (*str*)
    - **tensor**     (*Optional[torch.Tensor]*)
    - **persistent**     (*bool*)     – defaults to `True`

???- note "register_forward_hook"

    Register a forward hook on the module.

    The hook will be called every time after :func:`forward` has computed an output.  If ``with_kwargs`` is ``False`` or not specified, the input contains only the positional arguments given to the module. Keyword arguments won't be passed to the hooks and only to the ``forward``. The hook can modify the output. It can modify the input inplace but it will not have effect on forward since this is called after :func:`forward` is called. The hook should have the following signature::      hook(module, args, output) -> None or modified output  If ``with_kwargs`` is ``True``, the forward hook will be passed the ``kwargs`` given to the forward function and be expected to return the output possibly modified. The hook should have the following signature::      hook(module, args, kwargs, output) -> None or modified output  Args:     hook (Callable): The user defined hook to be registered.     prepend (bool): If ``True``, the provided ``hook`` will be fired         before all existing ``forward`` hooks on this         :class:`torch.nn.modules.Module`. Otherwise, the provided         ``hook`` will be fired after all existing ``forward`` hooks on         this :class:`torch.nn.modules.Module`. Note that global         ``forward`` hooks registered with         :func:`register_module_forward_hook` will fire before all hooks         registered by this method.         Default: ``False``     with_kwargs (bool): If ``True``, the ``hook`` will be passed the         kwargs given to the forward function.         Default: ``False``     always_call (bool): If ``True`` the ``hook`` will be run regardless of         whether an exception is raised while calling the Module.         Default: ``False``  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Union[Callable[[~T, Tuple[Any, ...], Any], Optional[Any]], Callable[[~T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]]]*)
    - **prepend**     (*bool*)     – defaults to `False`
    - **with_kwargs**     (*bool*)     – defaults to `False`
    - **always_call**     (*bool*)     – defaults to `False`

???- note "register_forward_pre_hook"

    Register a forward pre-hook on the module.

    The hook will be called every time before :func:`forward` is invoked.  If ``with_kwargs`` is false or not specified, the input contains only the positional arguments given to the module. Keyword arguments won't be passed to the hooks and only to the ``forward``. The hook can modify the input. User can either return a tuple or a single modified value in the hook. We will wrap the value into a tuple if a single value is returned (unless that value is already a tuple). The hook should have the following signature::      hook(module, args) -> None or modified input  If ``with_kwargs`` is true, the forward pre-hook will be passed the kwargs given to the forward function. And if the hook modifies the input, both the args and kwargs should be returned. The hook should have the following signature::      hook(module, args, kwargs) -> None or a tuple of modified input and kwargs  Args:     hook (Callable): The user defined hook to be registered.     prepend (bool): If true, the provided ``hook`` will be fired before         all existing ``forward_pre`` hooks on this         :class:`torch.nn.modules.Module`. Otherwise, the provided         ``hook`` will be fired after all existing ``forward_pre`` hooks         on this :class:`torch.nn.modules.Module`. Note that global         ``forward_pre`` hooks registered with         :func:`register_module_forward_pre_hook` will fire before all         hooks registered by this method.         Default: ``False``     with_kwargs (bool): If true, the ``hook`` will be passed the kwargs         given to the forward function.         Default: ``False``  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Union[Callable[[~T, Tuple[Any, ...]], Optional[Any]], Callable[[~T, Tuple[Any, ...], Dict[str, Any]], Optional[Tuple[Any, Dict[str, Any]]]]]*)
    - **prepend**     (*bool*)     – defaults to `False`
    - **with_kwargs**     (*bool*)     – defaults to `False`

???- note "register_full_backward_hook"

    Register a backward hook on the module.

    The hook will be called every time the gradients with respect to a module are computed, i.e. the hook will execute if and only if the gradients with respect to module outputs are computed. The hook should have the following signature::      hook(module, grad_input, grad_output) -> tuple(Tensor) or None  The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients with respect to the inputs and outputs respectively. The hook should not modify its arguments, but it can optionally return a new gradient with respect to the input that will be used in place of :attr:`grad_input` in subsequent computations. :attr:`grad_input` will only correspond to the inputs given as positional arguments and all kwarg arguments are ignored. Entries in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor arguments.  For technical reasons, when this hook is applied to a Module, its forward function will receive a view of each Tensor passed to the Module. Similarly the caller will receive a view of each Tensor returned by the Module's forward function.  .. warning ::     Modifying inputs or outputs inplace is not allowed when using backward hooks and     will raise an error.  Args:     hook (Callable): The user-defined hook to be registered.     prepend (bool): If true, the provided ``hook`` will be fired before         all existing ``backward`` hooks on this         :class:`torch.nn.modules.Module`. Otherwise, the provided         ``hook`` will be fired after all existing ``backward`` hooks on         this :class:`torch.nn.modules.Module`. Note that global         ``backward`` hooks registered with         :func:`register_module_full_backward_hook` will fire before         all hooks registered by this method.  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Callable[[ForwardRef('Module'), Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, Tuple[torch.Tensor, ...], torch.Tensor]]*)
    - **prepend**     (*bool*)     – defaults to `False`

???- note "register_full_backward_pre_hook"

    Register a backward pre-hook on the module.

    The hook will be called every time the gradients for the module are computed. The hook should have the following signature::      hook(module, grad_output) -> tuple[Tensor] or None  The :attr:`grad_output` is a tuple. The hook should not modify its arguments, but it can optionally return a new gradient with respect to the output that will be used in place of :attr:`grad_output` in subsequent computations. Entries in :attr:`grad_output` will be ``None`` for all non-Tensor arguments.  For technical reasons, when this hook is applied to a Module, its forward function will receive a view of each Tensor passed to the Module. Similarly the caller will receive a view of each Tensor returned by the Module's forward function.  .. warning ::     Modifying inputs inplace is not allowed when using backward hooks and     will raise an error.  Args:     hook (Callable): The user-defined hook to be registered.     prepend (bool): If true, the provided ``hook`` will be fired before         all existing ``backward_pre`` hooks on this         :class:`torch.nn.modules.Module`. Otherwise, the provided         ``hook`` will be fired after all existing ``backward_pre`` hooks         on this :class:`torch.nn.modules.Module`. Note that global         ``backward_pre`` hooks registered with         :func:`register_module_full_backward_pre_hook` will fire before         all hooks registered by this method.  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**     (*Callable[[ForwardRef('Module'), Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, Tuple[torch.Tensor, ...], torch.Tensor]]*)
    - **prepend**     (*bool*)     – defaults to `False`

???- note "register_load_state_dict_post_hook"

    Register a post hook to be run after module's ``load_state_dict`` is called.

    It should have the following signature::     hook(module, incompatible_keys) -> None  The ``module`` argument is the current module that this hook is registered on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys`` is a ``list`` of ``str`` containing the missing keys and ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.  The given incompatible_keys can be modified inplace if needed.  Note that the checks performed when calling :func:`load_state_dict` with ``strict=True`` are affected by modifications the hook makes to ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either set of keys will result in an error being thrown when ``strict=True``, and clearing out both missing and unexpected keys will avoid an error.  Returns:     :class:`torch.utils.hooks.RemovableHandle`:         a handle that can be used to remove the added hook by calling         ``handle.remove()``

    **Parameters**

    - **hook**

???- note "register_module"

    Alias for :func:`add_module`.

    **Parameters**

    - **name**     (*str*)
    - **module**     (*Optional[ForwardRef('Module')]*)

???- note "register_parameter"

    Add a parameter to the module.

    The parameter can be accessed as an attribute using given name.  Args:     name (str): name of the parameter. The parameter can be accessed         from this module using the given name     param (Parameter or None): parameter to be added to the module. If         ``None``, then operations that run on parameters, such as :attr:`cuda`,         are ignored. If ``None``, the parameter is **not** included in the         module's :attr:`state_dict`.

    **Parameters**

    - **name**     (*str*)
    - **param**     (*Optional[torch.nn.parameter.Parameter]*)

???- note "register_state_dict_pre_hook"

    Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

    These hooks will be called with arguments: ``self``, ``prefix``, and ``keep_vars`` before calling ``state_dict`` on ``self``. The registered hooks can be used to perform pre-processing before the ``state_dict`` call is made.

    **Parameters**

    - **hook**

???- note "requires_grad_"

    Change if autograd should record operations on parameters in this module.

    This method sets the parameters' :attr:`requires_grad` attributes in-place.  This method is helpful for freezing part of the module for finetuning or training parts of a model individually (e.g., GAN training).  See :ref:`locally-disable-grad-doc` for a comparison between `.requires_grad_()` and several similar mechanisms that may be confused with it.  Args:     requires_grad (bool): whether autograd should record operations on                           parameters in this module. Default: ``True``.  Returns:     Module: self

    **Parameters**

    - **requires_grad**     (*bool*)     – defaults to `True`

???- note "set_extra_state"

    Set extra state contained in the loaded `state_dict`.

    This function is called from :func:`load_state_dict` to handle any extra state found within the `state_dict`. Implement this function and a corresponding :func:`get_extra_state` for your module if you need to store extra state within its `state_dict`.  Args:     state (dict): Extra state from the `state_dict`

    **Parameters**

    - **state**     (*Any*)

???- note "share_memory"

    See :meth:`torch.Tensor.share_memory_`.


???- note "state_dict"

    Return a dictionary containing references to the whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are included. Keys are corresponding parameter and buffer names. Parameters and buffers set to ``None`` are not included.  .. note::     The returned object is a shallow copy. It contains references     to the module's parameters and buffers.  .. warning::     Currently ``state_dict()`` also accepts positional arguments for     ``destination``, ``prefix`` and ``keep_vars`` in order. However,     this is being deprecated and keyword arguments will be enforced in     future releases.  .. warning::     Please avoid the use of argument ``destination`` as it is not     designed for end-users.  Args:     destination (dict, optional): If provided, the state of module will         be updated into the dict and the same object is returned.         Otherwise, an ``OrderedDict`` will be created and returned.         Default: ``None``.     prefix (str, optional): a prefix added to parameter and buffer         names to compose the keys in state_dict. Default: ``''``.     keep_vars (bool, optional): by default the :class:`~torch.Tensor` s         returned in the state dict are detached from autograd. If it's         set to ``True``, detaching will not be performed.         Default: ``False``.  Returns:     dict:         a dictionary containing a whole state of the module  Example::      >>> # xdoctest: +SKIP("undefined vars")     >>> module.state_dict().keys()     ['bias', 'weight']

    **Parameters**

    - **args**
    - **destination**     – defaults to `None`
    - **prefix**     – defaults to ``
    - **keep_vars**     – defaults to `False`

???- note "to"

    Move and/or cast the parameters and buffers.

    This can be called as  .. function:: to(device=None, dtype=None, non_blocking=False)    :noindex:  .. function:: to(dtype, non_blocking=False)    :noindex:  .. function:: to(tensor, non_blocking=False)    :noindex:  .. function:: to(memory_format=torch.channels_last)    :noindex:  Its signature is similar to :meth:`torch.Tensor.to`, but only accepts floating point or complex :attr:`dtype`\ s. In addition, this method will only cast the floating point or complex parameters and buffers to :attr:`dtype` (if given). The integral parameters and buffers will be moved :attr:`device`, if that is given, but with dtypes unchanged. When :attr:`non_blocking` is set, it tries to convert/move asynchronously with respect to the host if possible, e.g., moving CPU Tensors with pinned memory to CUDA devices.  See below for examples.  .. note::     This method modifies the module in-place.  Args:     device (:class:`torch.device`): the desired device of the parameters         and buffers in this module     dtype (:class:`torch.dtype`): the desired floating point or complex dtype of         the parameters and buffers in this module     tensor (torch.Tensor): Tensor whose dtype and device are the desired         dtype and device for all parameters and buffers in this module     memory_format (:class:`torch.memory_format`): the desired memory         format for 4D parameters and buffers in this module (keyword         only argument)  Returns:     Module: self  Examples::      >>> # xdoctest: +IGNORE_WANT("non-deterministic")     >>> linear = nn.Linear(2, 2)     >>> linear.weight     Parameter containing:     tensor([[ 0.1913, -0.3420],             [-0.5113, -0.2325]])     >>> linear.to(torch.double)     Linear(in_features=2, out_features=2, bias=True)     >>> linear.weight     Parameter containing:     tensor([[ 0.1913, -0.3420],             [-0.5113, -0.2325]], dtype=torch.float64)     >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)     >>> gpu1 = torch.device("cuda:1")     >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)     Linear(in_features=2, out_features=2, bias=True)     >>> linear.weight     Parameter containing:     tensor([[ 0.1914, -0.3420],             [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')     >>> cpu = torch.device("cpu")     >>> linear.to(cpu)     Linear(in_features=2, out_features=2, bias=True)     >>> linear.weight     Parameter containing:     tensor([[ 0.1914, -0.3420],             [-0.5112, -0.2324]], dtype=torch.float16)      >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)     >>> linear.weight     Parameter containing:     tensor([[ 0.3741+0.j,  0.2382+0.j],             [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)     >>> linear(torch.ones(3, 2, dtype=torch.cdouble))     tensor([[0.6122+0.j, 0.1150+0.j],             [0.6122+0.j, 0.1150+0.j],             [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

    **Parameters**

    - **args**
    - **kwargs**

???- note "to_empty"

    Move the parameters and buffers to the specified device without copying storage.

    Args:     device (:class:`torch.device`): The desired device of the parameters         and buffers in this module.     recurse (bool): Whether parameters and buffers of submodules should         be recursively moved to the specified device.  Returns:     Module: self

    **Parameters**

    - **device**     (*Union[int, str, torch.device, NoneType]*)
    - **recurse**     (*bool*)     – defaults to `True`

???- note "train"

    Set the module in training mode.

    This has any effect only on certain modules. See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`, etc.  Args:     mode (bool): whether to set training mode (``True``) or evaluation                  mode (``False``). Default: ``True``.  Returns:     Module: self

    **Parameters**

    - **mode**     (*bool*)     – defaults to `True`

???- note "type"

    Casts all parameters and buffers to :attr:`dst_type`.

    .. note::     This method modifies the module in-place.  Args:     dst_type (type or string): the desired type  Returns:     Module: self

    **Parameters**

    - **dst_type**     (*Union[torch.dtype, str]*)

???- note "xpu"

    Move all model parameters and buffers to the XPU.

    This also makes associated parameters and buffers different objects. So it should be called before constructing optimizer if the module will live on XPU while being optimized.  .. note::     This method modifies the module in-place.  Arguments:     device (int, optional): if specified, all parameters will be         copied to that device  Returns:     Module: self

    **Parameters**

    - **device**     (*Union[int, torch.device, NoneType]*)     – defaults to `None`

???- note "zero_grad"

    Reset gradients of all model parameters.

    See similar function under :class:`torch.optim.Optimizer` for more context.  Args:     set_to_none (bool): instead of setting to zero, set the grads to None.         See :meth:`torch.optim.Optimizer.zero_grad` for details.

    **Parameters**

    - **set_to_none**     (*bool*)     – defaults to `True`
