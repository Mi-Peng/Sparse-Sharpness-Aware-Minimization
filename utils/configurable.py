import inspect
import functools
import argparse


def configurable(init_func=None, *, from_config=None):
    if init_func is not None:  # use for `__init__` in a class
        
        assert (
            inspect.isfunction(init_func)
            and from_config is None
            and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable."

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError("Class with @configurable must have a `from_config` classmethod.") from e
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a `from_config` classmethod.")
            
            if _called_with_args(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)
        return wrapped

    else:  # use for a function
        if from_config is None:
            return configurable
        assert inspect.isfunction(
            from_config
        ), "`from_config` argument of configurable must be a function."

        def wrapper(orig_func):
            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_args(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)
            
            wrapped.from_config = from_config
            return wrapped
        return wrapper

def _called_with_args(*args, **kwargs):
    '''
        Check whether the arguments contain `argparse`
    '''
    if len(args) and isinstance(args[0], (argparse.Namespace,)): # `from_config`'s first argument is forces to be `argparse`
        return True
    if isinstance(kwargs.pop("args", None), (argparse.Namespace,)):
        return True
    return False


def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.
    Returns:
        dict: arguments to be used for cls.__init__
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "args":
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take 'args' as the first argument!")
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg:  # forward all arguments to from_config, if from_config accepts them
        ret = from_config_func(*args, **kwargs)
    else:
        # forward supported arguments to from_config
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        # forward the other arguments to __init__
        ret.update(extra_kwargs)
    return ret
