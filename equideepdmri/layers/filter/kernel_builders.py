import re
from functools import partial
from typing import Union, List, Tuple

from equideepdmri.layers.filter.angular_basis_filters import SH_P_AngularKernelConstructor, AngularKernelBasisConstructor, \
    SH_Q_AngularKernelConstructor, SH_PQDiff_AngularKernelConstructor, TP_AngularKernelConstructor
from equideepdmri.layers.filter.combined_filter_kernels import SumKernel, ConcatKernel
from equideepdmri.layers.filter.filter_kernel import Kernel, KernelDefinitionInterface
from equideepdmri.layers.filter.radial_basis_functions import build_radial_basis_constructor
from equideepdmri.layers.filter.radial_basis_filters import LengthPDiff_ScalarKernelConstructor, RadialKernelBasisConstructor, \
    CombinedScalarKernelConstructor, LengthQOut_ScalarKernelConstructor, LengthQIn_ScalarKernelConstructor
from equideepdmri.layers.filter.utils import selection_rule_out, predefined_selection_rule_out, selection_rule, \
    predefined_selection_rule, SphericalTensorType, SelectionRuleInterface


def build_diff_kernel(normalization, normalize_Q_before_diff, sub_kernel_selection_rule,
                      has_Q_in, has_Q_out,
                      q_radial_basis_type='gaussian', q_radial_basis_params=None,
                      q_out_radial_basis_type=None, q_out_radial_basis_params=None,
                      q_in_radial_basis_type=None, q_in_radial_basis_params=None,
                      p_radial_basis_type='gaussian', p_radial_basis_params=None) \
        -> (AngularKernelBasisConstructor, RadialKernelBasisConstructor):
    assert has_Q_out or has_Q_in
    if not has_Q_out or not has_Q_in:
        normalize_Q_before_diff = False

    angular = SH_PQDiff_AngularKernelConstructor(normalization=normalization,
                                                 normalize_Q_before_diff=normalize_Q_before_diff)

    q_in_radial_constructor, q_out_radial_constructor = build_q_radial_basis_constructor(q_in_radial_basis_params,
                                                                                         q_in_radial_basis_type,
                                                                                         q_out_radial_basis_params,
                                                                                         q_out_radial_basis_type,
                                                                                         q_radial_basis_params,
                                                                                         q_radial_basis_type)

    p_radial_constructor = build_p_radial_basis_constructor(p_radial_basis_params, p_radial_basis_type)

    if has_Q_out and has_Q_in:
        scalar = CombinedScalarKernelConstructor(LengthPDiff_ScalarKernelConstructor(p_radial_constructor),
                                                 LengthQOut_ScalarKernelConstructor(q_out_radial_constructor),
                                                 LengthQIn_ScalarKernelConstructor(q_in_radial_constructor))
    elif has_Q_in:
        scalar = CombinedScalarKernelConstructor(LengthPDiff_ScalarKernelConstructor(p_radial_constructor),
                                                 LengthQIn_ScalarKernelConstructor(q_in_radial_constructor))
    else:
        scalar = CombinedScalarKernelConstructor(LengthPDiff_ScalarKernelConstructor(p_radial_constructor),
                                                 LengthQOut_ScalarKernelConstructor(q_out_radial_constructor))

    return angular, scalar


def build_tp_kernel(normalization, normalize_Q_before_diff, sub_kernel_selection_rule, has_Q_in, has_Q_out,
                    q_radial_basis_type='gaussian', q_radial_basis_params=None,
                    q_out_radial_basis_type=None, q_out_radial_basis_params=None,
                    q_in_radial_basis_type=None, q_in_radial_basis_params=None,
                    p_radial_basis_type='gaussian', p_radial_basis_params=None) \
        -> (AngularKernelBasisConstructor, RadialKernelBasisConstructor):
    assert has_Q_out or has_Q_in
    if not has_Q_out or not has_Q_in:
        normalize_Q_before_diff = False

    sub_kernel_selection_rule = _prepare_sub_kernel_selection_rule(sub_kernel_selection_rule)

    angular = TP_AngularKernelConstructor(SH_Q_AngularKernelConstructor(normalization=normalization,
                                                                        normalize_Q_before_diff=normalize_Q_before_diff),
                                          SH_P_AngularKernelConstructor(normalization=normalization),
                                          normalization=normalization,
                                          selection_rule=sub_kernel_selection_rule)

    q_in_radial_constructor, q_out_radial_constructor = build_q_radial_basis_constructor(q_in_radial_basis_params,
                                                                                         q_in_radial_basis_type,
                                                                                         q_out_radial_basis_params,
                                                                                         q_out_radial_basis_type,
                                                                                         q_radial_basis_params,
                                                                                         q_radial_basis_type)

    p_radial_constructor = build_p_radial_basis_constructor(p_radial_basis_params, p_radial_basis_type)

    if has_Q_out and has_Q_in:
        scalar = CombinedScalarKernelConstructor(LengthPDiff_ScalarKernelConstructor(p_radial_constructor),
                                                 LengthQOut_ScalarKernelConstructor(q_out_radial_constructor),
                                                 LengthQIn_ScalarKernelConstructor(q_in_radial_constructor))
    elif has_Q_in:
        scalar = CombinedScalarKernelConstructor(LengthPDiff_ScalarKernelConstructor(p_radial_constructor),
                                                 LengthQIn_ScalarKernelConstructor(q_in_radial_constructor))
    else:
        scalar = CombinedScalarKernelConstructor(LengthPDiff_ScalarKernelConstructor(p_radial_constructor),
                                                 LengthQOut_ScalarKernelConstructor(q_out_radial_constructor))

    return angular, scalar


def _prepare_sub_kernel_selection_rule(sub_kernel_selection_rule):
    if sub_kernel_selection_rule is None:
        # default selection rule
        sub_kernel_selection_rule = selection_rule_out()
    elif isinstance(sub_kernel_selection_rule, dict):
        if all(map(lambda k: isinstance(k, str) and not k.isdigit(), sub_kernel_selection_rule.keys())):
            # if map contains sting keys => interpret them as parameter for selection rule function
            sub_kernel_selection_rule = selection_rule_out(**sub_kernel_selection_rule)
        elif all(map(lambda k: isinstance(k, int) or (isinstance(k, str) and k.isdigit()),
                     sub_kernel_selection_rule.keys())):
            # if map contains int keys => interpret it as predefined selection rule where the keys are l_out
            sub_kernel_selection_rule = {
                int(k): v for k, v in sub_kernel_selection_rule.items()
            }
            sub_kernel_selection_rule = predefined_selection_rule_out(sub_kernel_selection_rule)
        else:
            raise ValueError(f'Invalid value for sub_kernel_selection_rule: '
                             f'Only two types of dicts can be given: '
                             f'int-keys (l_f) and List[Tuple[int, int]] values (l_f_1, l_f_2) for predefined_selection_rule_out '
                             f'or str-keys as parameters for selection_rule_out. '
                             f'But the given dict was: {sub_kernel_selection_rule}.')
    # else => directly use the sub_kernel_selection_rule as callable
    return sub_kernel_selection_rule


def build_p_space_kernel(normalization, normalize_Q_before_diff, sub_kernel_selection_rule,
                         has_Q_in, has_Q_out,
                         q_radial_basis_type=None, q_radial_basis_params=None,
                         q_out_radial_basis_type=None, q_out_radial_basis_params=None,
                         q_in_radial_basis_type=None, q_in_radial_basis_params=None,
                         p_radial_basis_type='gaussian', p_radial_basis_params=None) \
        -> (AngularKernelBasisConstructor, RadialKernelBasisConstructor):
    angular = SH_P_AngularKernelConstructor(normalization=normalization)

    p_radial_constructor = build_p_radial_basis_constructor(p_radial_basis_params, p_radial_basis_type)
    scalar = LengthPDiff_ScalarKernelConstructor(p_radial_constructor)

    return angular, scalar


def build_q_space_kernel(normalization, normalize_Q_before_diff, sub_kernel_selection_rule,
                         has_Q_in, has_Q_out,
                         q_radial_basis_type='gaussian', q_radial_basis_params=None,
                         q_out_radial_basis_type=None, q_out_radial_basis_params=None,
                         q_in_radial_basis_type=None, q_in_radial_basis_params=None,
                         p_radial_basis_type=None, p_radial_basis_params=None) \
        -> (AngularKernelBasisConstructor, RadialKernelBasisConstructor):
    assert has_Q_out or has_Q_in
    if not has_Q_out or not has_Q_in:
        normalize_Q_before_diff = False

    angular = SH_Q_AngularKernelConstructor(normalization=normalization,
                                            normalize_Q_before_diff=normalize_Q_before_diff)

    q_in_radial_constructor, q_out_radial_constructor = build_q_radial_basis_constructor(q_in_radial_basis_params,
                                                                                         q_in_radial_basis_type,
                                                                                         q_out_radial_basis_params,
                                                                                         q_out_radial_basis_type,
                                                                                         q_radial_basis_params,
                                                                                         q_radial_basis_type)

    if has_Q_out and has_Q_in:
        scalar = CombinedScalarKernelConstructor(LengthQOut_ScalarKernelConstructor(q_out_radial_constructor),
                                                 LengthQIn_ScalarKernelConstructor(q_in_radial_constructor))
    elif has_Q_in:
        scalar = LengthQIn_ScalarKernelConstructor(q_in_radial_constructor)
    else:
        scalar = LengthQOut_ScalarKernelConstructor(q_out_radial_constructor)

    return angular, scalar


_kernel_builders = {
    'pq_diff': build_diff_kernel,
    'pq_TP': build_tp_kernel,
    'p_space': build_p_space_kernel,
    'q_space': build_q_space_kernel,
}


def build_kernel_from_definition_name(kernel_definition_name: str, has_Q_in, has_Q_out, normalization='component',
                                      zero_length_eps: float = 1e-6, use_linear_model_for_zero_length=True,
                                      kernel_selection_rule=None,
                                      normalize_Q_before_diff=True,
                                      sub_kernel_selection_rule: dict = None,
                                      **kwargs) -> KernelDefinitionInterface:
    """
    :param kernel_definition_name:
    :param normalization:
    :param radial_basis_type:
    :param radial_basis_params:
    :param zero_length_eps:
    :param use_linear_model_for_zero_length:
    :param kernel_selection_rule:
    :param normalize_Q_before_diff: Only required for kernel types which contain q-space.
    :param sub_kernel_selection_rule: Only required for kernel type "pq_TP"
    :return:
    """
    assert kernel_definition_name in _kernel_builders, f'Unknown kernel definition name "{kernel_definition_name}", ' \
        f'supported values are {", ".join(_kernel_builders.keys())}, sum(<kernel_1>;<kernel_2>;...) and ' \
        f'concat([<out_type_1>]:<kernel_1>;[<out_type_2>]:<kernel_2>;...)'

    angular, scalar = _kernel_builders[kernel_definition_name](has_Q_in=has_Q_in, has_Q_out=has_Q_out,
                                                               normalization=normalization,
                                                               normalize_Q_before_diff=normalize_Q_before_diff,
                                                               sub_kernel_selection_rule=sub_kernel_selection_rule,
                                                               **kwargs)

    if kernel_selection_rule is None:
        kernel_selection_rule = selection_rule()
    elif isinstance(kernel_selection_rule, dict):
        if all(map(lambda k: isinstance(k, str), kernel_selection_rule.keys())):
            # if map contains sting keys => interpret them as parameter for selection rule function
            kernel_selection_rule = selection_rule(**kernel_selection_rule)
        elif all(map(lambda k: isinstance(k, tuple) or isinstance(k, list), kernel_selection_rule.keys())):
            # if map contains int keys => interpret it as predefined selection rule where the keys are l_out
            kernel_selection_rule = predefined_selection_rule(kernel_selection_rule)
        else:
            raise ValueError(f'Invalid value for kernel_selection_rule: '
                             f'Only two types of dicts can be given: '
                             f'Tuple[int, int]-keys (l_in, l_out) and List[int] values (possible l_f) for predefined_selection_rule '
                             f'or str-keys as parameters for selection_rule. '
                             f'But the given dict was: {kernel_selection_rule}.')
    # else => directly use the kernel_selection_rule as callable

    kernel = KernelDefinition(angular, scalar,
                              zero_length_eps=zero_length_eps,
                              use_linear_model_for_zero_length=use_linear_model_for_zero_length,
                              selection_rule=kernel_selection_rule,
                              normalization=normalization)

    return kernel


def _is_sum_kernel_definition(kernel_definition_name: str) -> bool:
    return re.fullmatch(r'sum\(\w+(;\w+)*\)', kernel_definition_name) is not None


def build_sum_kernel(kernel_definition_name: str, **kernel_kwargs) -> KernelDefinitionInterface:
    sub_kernel_definitions = re.fullmatch(r'sum\((.*)\)', kernel_definition_name).group(1).split(';')

    return SumKernelDefinition(*sub_kernel_definitions, **kernel_kwargs)


def _is_concat_kernel_definition(kernel_definition_name: str) -> bool:
    return re.fullmatch(r'concat\(\[\d+(,\d+)*\]:\w+(;\[\d+(,\d+)*\]:\w+)*\)', kernel_definition_name) is not None


def build_concat_kernel(kernel_definition_name: str, **kernel_kwargs) -> KernelDefinitionInterface:
    sub_kernel_definition_strings = re.fullmatch(r'concat\((.*)\)', kernel_definition_name).group(1).split(';')

    sub_kernel_definitions = []
    for kernel_def_string in iter(sub_kernel_definition_strings):
        kernel_out_type_str, definition_str = kernel_def_string.split(':')
        kernel_out_type = list(map(int, kernel_out_type_str.strip('][').split(',')))  # list of multiplicities (ints)

        sub_kernel_definitions.append((kernel_out_type, definition_str))

    return ConcatKernelDefinition(*sub_kernel_definitions, **kernel_kwargs)


def build_kernel(kernel_definition: Union[KernelDefinitionInterface, str], **kernel_kwargs) -> KernelDefinitionInterface:
    if isinstance(kernel_definition, str):
        if _is_sum_kernel_definition(kernel_definition):
            return build_sum_kernel(kernel_definition, **kernel_kwargs)
        elif _is_concat_kernel_definition(kernel_definition):
            return build_concat_kernel(kernel_definition, **kernel_kwargs)
        else:
            return build_kernel_from_definition_name(kernel_definition, **kernel_kwargs)
    else:
        assert len(kernel_kwargs) == 0, f'Unexpected parameters for kernel: {kernel_kwargs.keys()}. ' \
            f'Note that if kernel_definition is not a string then no parameters are accepted. ' \
            f'If using SumKernelDefinition or ConcatKernelDefinition, then the parameters for the kernels need to be given there.'
        return kernel_definition


def SumKernelDefinition(*kernel_definitions: List[Union[KernelDefinitionInterface, str]], **kernel_kwargs) \
        -> KernelDefinitionInterface:
    """
    Note: the kernel_kwargs are used for all kernel_definitions.
    To use different kernel_kwargs for the different kernel_definitions:
    use the function build_kernel(kernel_definition, **kernel_kwargs) for each of the kernel definitions instead of passing them directly.
    In this case no **kernel_kwargs must be passed directly to ConcatKernel_Constructor.

    :param kernel_definitions:
    :param kernel_kwargs:
    :return:
    """
    kernel_definitions = [build_kernel(kernel_definition, **kernel_kwargs) for kernel_definition in kernel_definitions]

    return partial(SumKernel, kernel_definitions=kernel_definitions)


def ConcatKernelDefinition(*kernel_definitions: List[Tuple[Union[SphericalTensorType, List[int]],
                                                           Union[KernelDefinitionInterface, str]]],
                           **kernel_kwargs) -> KernelDefinitionInterface:
    """
    Note: the kernel_kwargs are used for all kernel_definitions.
    To use different kernel_kwargs for the different kernel_definitions:
    use the function build_kernel(kernel_definition, **kernel_kwargs) for each of the kernel definitions instead of passing them directly.
    In this case no **kernel_kwargs must be passed directly to ConcatKernel_Constructor.

    :param kernel_definitions:
    :param kernel_kwargs:
    :return:
    """
    kernel_definitions = [(
        kernel_type_out
        if isinstance(kernel_type_out, SphericalTensorType)
        else SphericalTensorType.from_multiplicities(kernel_type_out),

        build_kernel(kernel_definition, **kernel_kwargs)
    ) for kernel_type_out, kernel_definition in kernel_definitions]

    return partial(ConcatKernel, kernel_definitions=kernel_definitions)


def KernelDefinition(angular_kernel_constructor: AngularKernelBasisConstructor,
                     scalar_kernel_constructor: RadialKernelBasisConstructor,
                     zero_length_eps: float = 1e-6,
                     selection_rule: SelectionRuleInterface = selection_rule(),
                     use_linear_model_for_zero_length=True,
                     normalization='component') -> KernelDefinitionInterface:
    return partial(Kernel,
                   angular_kernel_constructor=angular_kernel_constructor,
                   scalar_kernel_constructor=scalar_kernel_constructor,
                   zero_length_eps=zero_length_eps,
                   selection_rule=selection_rule,
                   use_linear_model_for_zero_length=use_linear_model_for_zero_length,
                   normalization=normalization)

def build_p_radial_basis_constructor(p_radial_basis_params, p_radial_basis_type):
    if p_radial_basis_params is None:
        p_radial_basis_params = {}
    p_radial_constructor = build_radial_basis_constructor(p_radial_basis_type, **p_radial_basis_params)
    return p_radial_constructor


def build_q_radial_basis_constructor(q_in_radial_basis_params, q_in_radial_basis_type, q_out_radial_basis_params,
                                     q_out_radial_basis_type, q_radial_basis_params, q_radial_basis_type):
    if q_radial_basis_params is None:
        q_radial_basis_params = {}
    q_out_radial_basis_type = q_out_radial_basis_type if q_out_radial_basis_type is not None else q_radial_basis_type
    q_out_radial_basis_params = dict(q_radial_basis_params, **q_out_radial_basis_params) \
        if q_out_radial_basis_params is not None else q_radial_basis_params
    q_in_radial_basis_type = q_in_radial_basis_type if q_in_radial_basis_type is not None else q_radial_basis_type
    q_in_radial_basis_params = dict(q_radial_basis_params, **q_in_radial_basis_params) \
        if q_in_radial_basis_params is not None else q_radial_basis_params
    q_out_radial_constructor = build_radial_basis_constructor(q_out_radial_basis_type, **q_out_radial_basis_params)
    q_in_radial_constructor = build_radial_basis_constructor(q_in_radial_basis_type, **q_in_radial_basis_params)
    return q_in_radial_constructor, q_out_radial_constructor
