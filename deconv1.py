# based on the KellyKapowski interface
from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import range, str, bytes

import os
import warnings
import sys
import re
import nibabel as nb
import numpy as np

from nipype.utils.filemanip import split_filename

from nipype.interfaces.base import (BaseInterface, TraitedSpec, traits, File, OutputMultiPath,
                                    BaseInterfaceInputSpec, isdefined,
                                    CommandLineInputSpec, CommandLine, StdOutCommandLine, StdOutCommandLineInputSpec,
                                    InputMultiPath, Directory, Undefined)

from nipype.interfaces.afni.base import (
    AFNICommand, AFNICommandBase, AFNICommandInputSpec,
    AFNICommandOutputSpec)


class DeconInputSpec(CommandLineInputSpec):
    # TODO: Add position metadata. Check if better using traits.Enum for local and global options
    timing = traits.Enum(
        'local', 'global',
        desc='Specify if stimulus timings in stim_files are \'local\' or \'global\'',
        argstr='-%s_times \\\n',
        position=1,
        usedefault=True
    )

    stop = traits.Bool(
        desc='stop after generating the design matrix',
        argstr='-x1D_stop \\\n',
        position=2
    )

    goforit = traits.Int(
        desc='maximum number of warnings before stopping execution',
        argstr='-GOFORIT %d \\\n',
        position=3
    )

    polort = traits.Either(
        traits.Enum('A'), traits.Int(),
        desc='order of baseline polynomial',
        argstr='-polort %s \\\n',
        default='A',
        usedefault=True,
        position=4
    )

    output_datatype = traits.Enum(
        'float', 'short',
        desc='Specify output datasets format. Valid types are \'float\' and \'short\'',
        argstr='-%s \\\n',
        position=5,
        usedefault=True
    )

    # TODO: check if it can be done with traits.List or traits.Dict (to include labels). Add position metadata
    # change name of in_file attribut --> input_files
    in_file = InputMultiPath(
        File(
            exists=True
        ),
        desc='hola',
        argstr='-input %s \\\n',
        mandatory=True,
        copyfile=False
    )

    num_stimts = traits.Int(
        desc='number of stim time files',
        argstr='-num_stimts %d \\\n',
        mandatory=True
    )

    stim_files = InputMultiPath(
        File(
            exists=True
        ),
        desc='List of containing onset files',
        mandatory=True,
        requires=['models', 'labels']
    )

    models = traits.List(
        traits.Str,
        desc='List models for each stim_times',
        minlen=1
    )

    labels = traits.List(
        traits.Str,
        desc='List of labels for the stimuli. \
        Must be sorted as in stim_files and models',
        minlen=1
    )

    ortvec = File(
        desc='Motion regressor file',
        argstr='-ortvec %s',
        exists=True
    )

    out_xmat = File('X.xmat.1D',
                    desc='name of output design matrix',
                    argstr='-x1D %s \n',
                    usedefault=True,
                    genfile=True,
                    position=-1
                    )

    out_xjpeg = File(
        desc='generate image of design matrix',
        name_template='%s.jpg \\\n',
        name_source='out_xmat',
        keep_extension=False,
        argstr='-xjpeg %s'
    )
    # bucket = traits.Undefined()
    # cbucket
    # rout
    fout = traits.Bool(
        desc='output F-statistics for each stimulus',
        argstr='-fout \\\n'
    )

    rout = traits.Bool(
        desc='output R^2 statistics',
        argstr='-rout \\\n'
    )

    tout = traits.Bool(
        desc='output T-statistics for each stimulus',
        argstr='-tout \\\n'
    )

    vout = traits.Bool(
        desc='output the sample variance (MSE) map',
        argstr='-vout \\\n'
    )

    bout = traits.Bool(
        desc='turn on output of baseline coefs and stats',
        argstr='-bout \\\n'
    )

    # no_cout = traits.Bool(
    #     desc='supress output of regression coefficients (and associated statistics)',
    #     argstr='-nocout \\\n'
    # )

    no_bucket = traits.Bool(
        False,
        desc='do not create output bucket',
        argstr='-nobucket',
        usedefault=True,
        position=-2
    )
    # out_cbucket xor no_bucket
    # xsave bool, requires -bucket, named by bucket

    out_file = File('Decon.nii.gz',
                    desc='name of output bucket',
                    argstr='-bucket %s \\\n',
                    genfile=True,
                    usedefault=True
                    )


class DeconOutputSpec(TraitedSpec):
    out_xmat = File(
        desc='X mat',
        exists=True
    )
    # out_bucket = File(
    #     desc='output statistics'
    # )
    out_file = File(
        desc='output statistics'
    )


class Decon(CommandLine):
    # class Decon(AFNICommand):

    _cmd = '3dDeconvolve'
    input_spec = DeconInputSpec
    output_spec = DeconOutputSpec

    def _format_arg(self, name, trait_spec, value):

        if name == 'out_xmat':
            xmat = self._gen_filename('out_xmat')
            return trait_spec.argstr % xmat

        if name == 'out_file' and not self.inputs.no_bucket:
            bucket = self._gen_filename('out_file')
            return trait_spec.argstr % bucket

        if name == 'num_stimts':
            arg = trait_spec.argstr % value
            N = self.inputs.num_stimts
            sf = self.inputs.stim_files
            num = range(1,N+1)
            sm = self.inputs.models
            sl = self.inputs.labels

            arg_aux = ' '.join('-stim_times %s %s \'%s\' -stim_label %s %s \\\n' % z for z in zip(num, sf, sm, num, sl))
            arg = ' '.join([arg,arg_aux])
            return arg

        return super(Decon, self)._format_arg(name, trait_spec, value)

    def _gen_filename(self, name):
        if name == 'out_xmat':
            output = self.inputs.out_xmat
            _, filename, ext = split_filename(output)
            if filename.endswith('xmat'):
                output = filename + '.1D'
            else:
                output = filename + '.xmat.1D'
            return output

        if name == 'out_file':
            output = self.inputs.out_file
            _, filename, ext = split_filename(output)
            output = filename + '_stats.nii.gz'
            return output

        return None

    def _parse_inputs(self, skip=None):
        # Skip the arguments without argstr metadata
        if skip is None:
            skip = []
        skip += ['stim_files', 'labels', 'models']

        # Skip output bucket if no_bucket == True
        if self.inputs.no_bucket:
            skip += ['out_file']

        return super(Decon, self)._parse_inputs(skip=skip)


    def _list_outputs(self):
        outputs = self.output_spec().get()
        # outputs['out_file'] = 'hola.nii'

# command to output