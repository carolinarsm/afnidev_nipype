#from __future__ import print_function, division, unicode_literals, absolute_import
# from builtins import open

from __future__ import print_function, unicode_literals
from builtins import object, str, bytes, open

# Stdlib imports
import os
import re
import sys
import warnings

import os
import re

from nipype.interfaces.afni.base import (
    AFNICommand, AFNICommandBase, AFNICommandInputSpec,
    AFNICommandOutputSpec)
from nipype.interfaces.base import (
    CommandLineInputSpec, CommandLine, StdOutCommandLine, StdOutCommandLineInputSpec,
    traits, TraitedSpec, File, Str, InputMultiPath, isdefined
)


class DeconInputSpec(AFNICommandInputSpec):

    # TODO: Add position metadata. Check if better using traits.Enum for local and global options
    local_times = traits.Bool(
        desc='True if stim time files are local',
        argstr='-local_times'
    )

    stop = traits.Bool(
        desc='stop after generating the design matrix',
        argstr='x1D_stop'
    )

    goforit = traits.Int(
        desc='maximum number of warnings before stopping execution',
        argstr='-GOFORIT %d'
    )

    # TODO: check if can be done with traits. Either, with default value 'A'
    polort = traits.Either(
        traits.Enum('A'), traits.Int(),
        desc='order of baseline polynomial',
        argstr='-polort %s',
        mandatory=True,
        default='A',
        usedefault=True
    )

    output_datatype = traits.Enum(
        'float', 'short',
        desc='Specify output datasets format. Valid types are \'float\' and \'short\'',
        argstr='-%s',
    )

    # TODO: check if it can be done with traits.List or traits.Dict (to include labels). Add position metadata
    in_file = InputMultiPath(
        File(
            exists=True
        ),
        desc='hola',
        argstr='-input %s',
        mandatory=True,
        copyfile=False
    )
    # in_files_LIST = traits.List(
    #     File(
    #         exists=True),
    #     minlen=1,
    #     sep=' ',
    #     argstr='-input \'%s\'',
    #     position=2,
    #     mandatory=True,
    #     desc="Input(s) file(s) to 3dDeconvolve.",
    #     copyfile=False
    # )
    #
    # in_file_TUP = traits.Tuple(
    #     minlen=1,
    #     argstr='-input \'%s\'',
    #     position=2,
    #     mandatory=True,
    #     desc="Input(s) file(s) to 3dDeconvolve.",
    # )

    num_stimts = traits.Int(
        desc='number of stim time files',
        argstr='-num_stimts %d',
        mandatory=True
    )

    # regressors inputs
    # TODO: check if it can be done with traits.Dict, to have stim times, stim labels and model?
    # TODO: Tuples?
    # stim_files = traits.Tuple(File(),
    #                           desc='Tuple containing the sorted onset files',
    #                           argstr='-stim_times %s',
    #                           exists=True,
    #                           requires=['models','labels']
    #                           )
    stim_files = InputMultiPath(
        File(
            exists=True
        ),
        desc='List of containing onset files',
        mandatory=True,
        requires=['models', 'labels']
    )

    # List? Tuple?
    models = traits.List(
        traits.Str(),
        desc='List models for each stim_times',
        minlen=1,
        mandatory=True,
        requires=['stim_files', 'labels']
    )
    # models = traits.Tuple(
    #     desc='Tuple containing models for each stim_times',
    #     mandatory=True,
    #     requires=['stim_files', 'labels']
    # )

    # List? Tuple?

    labels = traits.List(
        traits.Str(),
        desc='List of labels for the stimuli. \
        Must be sorted as in stim_files and models',
        minlen=1,
        mandatory=True,
        requires=['stim_files', 'models']
    )
    # labels = traits.Tuple(
    #     desc='Tuple containing labels for the stimulus. Must be sorted as in stim_files and models',
    #     mandatory=True,
    #     requires=['stim_files', 'models']
    # )

    ortvec = File(
        desc='Motion regressor file',
        argstr='-ortvec %s',
        exists=True
    )

    # TODO: is %s in argstr correct?. Is it mandatory and does it need a default value?
    xmat = File(
        desc='prefix for design matrix',
        argstr='-x1D %s',
        mandatory=True,
        name_template='%s.xmat.1D',
        default='X.xmat.1D',
        usedefault=True
    )

    #    xjpeg = traits.Undefined()

    #    bucket = traits.Undefined()
    # cbucket
    # rout

    nobucket = traits.Bool(
        desc='do not create output bucket',
        argstr='-nobucket',
        xor=['bucket']
    )

    fout = traits.Bool(
        desc='add F-stats to output bucket',
        argstr='-fout'
    )

    tout = traits.Bool(
        desc='add T-stats to output bucket',
        argstr='-fout'
    )


class DeconOutputSpec(TraitedSpec):
    xmat = File(
        # exists=True,
        desc='design matrix',
    )


class Decon(AFNICommand):

    _cmd = '3dDeconvolve'
    input_spec = DeconInputSpec
    output_spec = DeconOutputSpec

    # # def _format_arg(self, name, trait_spec, value):
    # # def _filename_from_source(self, name, chain=None):
    # # def _gen_filename(self, name):
    # # def _overload_extension(self, value, name=None):
    # # def _list_outputs(self):
    # # def _parse_inputs(self, skip=None):
    #
    # def _list_outputs(self):
    #     outputs = self.output_spec().get() # xmat, bucket
    #     for k in list(outputs.keys()):
    #         if k not in ('outputtype', 'environ', 'args'):
    #             if k != 'tensor' or (isdefined(self.inputs.save_tensor) and
    #                                      self.inputs.save_tensor):
    #                 outputs[k] = self._gen_fname(
    #                     self.inputs.base_name, suffix='_' + k)
    #                 return outputs
    #
    # def _format_arg(self, name, trait_spec, value):
    #     Undefined
    #

    def _list_outputs(self):
        outputs = self.output_spec().get()

        if not isdefined(self.inputs.xmat):
            outputs['xmat'] = self._gen_filename(self.inputs.xmat)
        else:
            outputs['xmat'] = os.path.abspath(self.inputs.xmat)

        # if isdefined(self.inputs.design):
        #     outputs['design'] = os.path.abspath(self.inputs.design)
        #
        # if isdefined(self.inputs.img_file):
        #     outputs['img_file'] = os.path.abspath(self.inputs.img_file)

        return outputs

    # def _format_arg(self, name, trait_spec, value):
    #     if name == 'num_stimts':
    #
    #         argN = trait_spec.argstr % value
    #         print argN
    #         n = len(self.inputs.stim_files)
    #         print n
    #         sf = self.inputs.stim_files
    #         sl = self.inputs.labels
    #         mod = self.inputs.models
    #         num = range(1, n+1)
    #         print num
    #         argT = ('-stim_times %s %s \'%s\' -labels %s %s\n' % z for z in zip(num, sf, mod, num, sl))
    #         # arg = ' '.join([trait_spec.argstr % z for z in zip(num, sf, mod, num, sl)])
    #         arg = argN + '\n' + argT
    #         return arg
    #         # elif name == 'glms':
    #         #     n = len(self.inputs.glm_contrasts)
    #         #     gc = self.inputs.glm_contrasts
    #         #     gl = self.inputs.glm_labels
    #         #     num = range(1, n + 1)
    #         #     arg = ' '.join([trait_spec.argstr % z for z in zip(gc, num, gl)])
    #         #     return arg
    #     return super(Decon, self)._format_arg(name, trait_spec, value)
    def _format_arg(self, name, trait_spec, value):
            if name == 'num_stimts':

                arg = trait_spec.argstr % value

                for i in range(0, self.inputs.num_stimts):

                    if isdefined(self.inputs.stim_files) :
                        arg += '-stim_files %s %s' %(i+1, self.inputs.stim_files[i])

                    if isdefined(self.inputs.models):
                        arg += '\'%s\'' % (self.inputs.models[i])

                    if isdefined(self.inputs.labels):
                        arg += '-labels %s %s\n' % (i+1,self.inputs.labels[i])

                return arg
            return super(Decon, self)._format_arg(name, trait_spec, value)







        # def _gen_filename(self, name):
        #     if name == 'xmat':
        #         return self._list_outputs()[name]


        #
        # def _list_outputs(self):
        #     outputs = self.output_spec().get()
        #     if not isdefined(self.inputs.out_file):
        #         outputs['out_file'] = self._gen_filename(self.inputs.out_file)
        #     else:
        #         outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        #     #        if isdefined(self.inputs.img_file):
        #     #            outputs['img_file'] = os.path.abspath(self.inputs.img_file)
        #     if isdefined(self.inputs.bucket):
        #         outputs['bucket'] = os.path.abspath(self.inputs.bucket)
        #
        #     return outputs
        #
        # def _gen_filename(self, name):
        #     if name == 'out_file':
        #         return self._list_outputs()[name]

        # def aggregate_outputs(self, runtime=None, needed_outputs=None):
        #    outputs = self._outputs()
        #    outfile = os.path.abspath(outputs.out_file + '.xmat.1D')
        #    outputs['out_file'] = outfile

        #    return outputs


# 3dinfo to get number of volumes
# Can I get the stdout as a Str without passing through a file?

# BREAK
#########
#########




