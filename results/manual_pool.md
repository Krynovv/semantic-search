# Пул функций для ручных запросов

Источник: `/mnt/proj/proj/study/study-ensemble/kursovaya-tex/code/var/data/test_pool.jsonl` (тест-корпус, 5000 функций).
Выбрано 150 случайных функций (seed=42).

Для каждой функции автор пишет русский запрос в `data/manual_queries.jsonl`:
```json
{"id": "<id функции>", "query_ru": "<запрос на русском>"}
```

## test-20659 — record_magic

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/IPython/core/magic.py#L118-L138
- repo: cloud9ers/gurumate

```python
def record_magic(dct, magic_kind, magic_name, func):
    















    if magic_kind == 'line_cell':
        dct['line'][magic_name] = dct['cell'][magic_name] = func
    else:
        dct[magic_kind][magic_name] = func
```

## test-20726 — mappable

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/IPython/parallel/client/map.py#L159-L166
- repo: cloud9ers/gurumate

```python
def mappable(obj):
    
    if isinstance(obj, (tuple,list)):
        return True
    for m in arrayModules:
        if isinstance(obj,m['type']):
            return True
    return False
```

## test-1294 — BatchNormalization._validate_bn_layer

- url: https://github.com/tensorflow/probability/blob/e87fe34111d68c35db0f9eeb4935f1ece9e1a8f5/tensorflow_probability/python/bijectors/batch_normalization.py#L164-L182
- repo: tensorflow/probability

```python
def _validate_bn_layer(self, layer):
    








    if (not isinstance(layer, tf.keras.layers.BatchNormalization) and
        not isinstance(layer, tf.compat.v1.layers.BatchNormalization)):
      raise ValueError(
          "batchnorm_layer must be an instance of BatchNormalization layer.")
    if layer.renorm:
      raise ValueError("BatchNorm Bijector does not support renormalization.")
    if layer.virtual_batch_size:
      raise ValueError(
          "BatchNorm Bijector does not support virtual batch sizes.")
```

## test-13969 — Session.raise_server_error

- url: https://github.com/elliterate/capybara.py/blob/0c6ae449cc37e4445ec3cd6af95674533beedc6c/capybara/session.py#L646-L653
- repo: elliterate/capybara.py

```python
def raise_server_error(self):
        
        if self.server and self.server.error:
            try:
                if capybara.raise_server_errors:
                    raise self.server.error
            finally:
                self.server.reset_error()
```

## test-10371 — PathParser.p_ctx_path

- url: https://github.com/IEMLdev/ieml/blob/4c842ba7e6165e2f1b4a4e2e98759f9f33af5f25/ieml/grammar/paths/parser/parser.py#L65-L70
- repo: IEMLdev/ieml

```python
def p_ctx_path(self, p):
        
        if len(p[1]) == 1:
            p[0] = p[1][0]
        else:
            p[0] = ContextPath(p[1])
```

## test-15262 — Nodelist.remove_node

- url: https://github.com/buzzfeed/phonon/blob/32fd036d64fab19c554e841f162466f6eb28b50f/phonon/nodelist.py#L75-L84
- repo: buzzfeed/phonon

```python
def remove_node(self, node_id=None):
        




        if not node_id:
            node_id = self.conn.id

        self.conn.client.hdel(self.nodelist_key, node_id)
```

## test-10736 — parse_config_string

- url: https://github.com/urinieto/msaf/blob/9dbb57d77a1310465a65cc40f1641d083ca74385/msaf/configparser.py#L34-L56
- repo: urinieto/msaf

```python
def parse_config_string(config_string, issue_warnings=True):
    


    config_dict = {}
    my_splitter = shlex.shlex(config_string, posix=True)
    my_splitter.whitespace = ','
    my_splitter.whitespace_split = True
    for kv_pair in my_splitter:
        kv_pair = kv_pair.strip()
        if not kv_pair:
            continue
        kv_tuple = kv_pair.split('=', 1)
        if len(kv_tuple) == 1:
            if issue_warnings:
                MsafConfigWarning.warn(
                    ("Config key '%s' has no value, ignoring it" %
                     kv_tuple[0]), stacklevel=1)
        else:
            k, v = kv_tuple
            # subsequent values for k will override earlier ones
            config_dict[k] = v
    return config_dict
```

## test-15404 — send_confirmed_notifications

- url: https://github.com/zenodo/zenodo-accessrequests/blob/ce2cf3f1425d02ba4f3ad3202cfca43a1892558a/zenodo_accessrequests/receivers.py#L89-L114
- repo: zenodo/zenodo-accessrequests

```python
def send_confirmed_notifications(request):
    
    pid, record = get_record(request.recid)
    if record is None:
        current_app.logger.error("Cannot retrieve record %s. Emails not sent"
                                 % request.recid)
        return
    title = _("Access request: %(record)s", record=record["title"])

    _send_notification(
        request.receiver.email,
        title,
        "zenodo_accessrequests/emails/new_request.tpl",
        request=request,
        record=record,
        pid=pid,
    )

    _send_notification(
        request.sender_email,
        title,
        "zenodo_accessrequests/emails/confirmation.tpl",
        request=request,
        record=record,
        pid=pid,
    )
```

## test-7749 — Evaluations.get_evaluations

- url: https://github.com/jrxFive/python-nomad/blob/37df37e4de21e6f8ac41c6154e7f1f44f1800020/nomad/api/evaluations.py#L60-L73
- repo: jrxFive/python-nomad

```python
def get_evaluations(self, prefix=None):
        










        params = {"prefix": prefix}
        return self.request(method="get", params=params).json()
```

## test-5162 — RateLimitHandler.setup_rate_limit_handler

- url: https://github.com/chaoss/grimoirelab-perceval/blob/41c908605e88b7ebc3a536c643fa0f212eaf9e0e/perceval/client.py#L225-L248
- repo: chaoss/grimoirelab-perceval

```python
def setup_rate_limit_handler(self, sleep_for_rate=False, min_rate_to_sleep=MIN_RATE_LIMIT,
                                 rate_limit_header=RATE_LIMIT_HEADER,
                                 rate_limit_reset_header=RATE_LIMIT_RESET_HEADER):
        






        self.rate_limit = None
        self.rate_limit_reset_ts = None
        self.sleep_for_rate = sleep_for_rate
        self.rate_limit_header = rate_limit_header
        self.rate_limit_reset_header = rate_limit_reset_header

        if min_rate_to_sleep > self.MAX_RATE_LIMIT:
            msg = "Minimum rate to sleep value exceeded (%d)."
            msg += "High values might cause the client to sleep forever."
            msg += "Reset to %d."
            self.min_rate_to_sleep = self.MAX_RATE_LIMIT
            logger.warning(msg, min_rate_to_sleep, self.MAX_RATE_LIMIT)
        else:
            self.min_rate_to_sleep = min_rate_to_sleep
```

## test-8619 — replace

- url: https://github.com/fastmonkeys/stellar/blob/79f0353563c35fa6ae46a2f00886ab1dd31c4492/stellar/command.py#L171-L182
- repo: fastmonkeys/stellar

```python
def replace(name):
    
    app = get_app()

    snapshot = app.get_snapshot(name)
    if not snapshot:
        click.echo("Couldn't find snapshot %s" % name)
        sys.exit(1)

    app.remove_snapshot(snapshot)
    app.create_snapshot(name)
    click.echo("Replaced snapshot %s" % name)
```

## test-10013 — Client.get_bucket

- url: https://github.com/singularityhub/sregistry-cli/blob/abc96140a1d15b5e96d83432e1e0e1f4f8f36331/sregistry/main/s3/__init__.py#L70-L89
- repo: singularityhub/sregistry-cli

```python
def get_bucket(self):
        


        for attr in ['bucket_name', 's3']:
            if not hasattr(self, attr):
                bot.exit('client is missing attribute %s' %(attr))

        # See if the bucket is already existing
        self.bucket = None
        for bucket in self.s3.buckets.all():
            if bucket.name == self.bucket_name:
                self.bucket = bucket
 
        # If the bucket doesn't exist, create it
        if self.bucket is None:
            self.bucket = self.s3.create_bucket(Bucket=self.bucket_name)
            bot.info('Created bucket %s' % self.bucket.name )

        return self.bucket
```

## test-6176 — TrelloService.annotations

- url: https://github.com/ralphbean/bugwarrior/blob/b2a5108f7b40cb0c437509b64eaa28f941f7ac8b/bugwarrior/services/trello.py#L115-L122
- repo: ralphbean/bugwarrior

```python
def annotations(self, card_json):
        

        comments = self.get_comments(card_json['id'])
        annotations = self.build_annotations(
            ((c['memberCreator']['username'], c['data']['text']) for c in comments),
            card_json["shortUrl"])
        return annotations
```

## test-1034 — _prepare_args

- url: https://github.com/tensorflow/probability/blob/e87fe34111d68c35db0f9eeb4935f1ece9e1a8f5/tensorflow_probability/python/mcmc/slice_sampler_kernel.py#L583-L615
- repo: tensorflow/probability

```python
def _prepare_args(target_log_prob_fn, state, step_size,
                  target_log_prob=None, maybe_expand=False,
                  description='target_log_prob'):
  
  state_parts = list(state) if mcmc_util.is_list_like(state) else [state]
  state_parts = [
      tf.convert_to_tensor(value=s, name='current_state') for s in state_parts
  ]

  target_log_prob = _maybe_call_fn(
      target_log_prob_fn,
      state_parts,
      target_log_prob,
      description)
  step_sizes = (list(step_size) if mcmc_util.is_list_like(step_size)
                else [step_size])
  step_sizes = [
      tf.convert_to_tensor(
          value=s, name='step_size', dtype=target_log_prob.dtype)
      for s in step_sizes
  ]
  if len(step_sizes) == 1:
    step_sizes *= len(state_parts)
  if len(state_parts) != len(step_sizes):
    raise ValueError('There should be exactly one `step_size` or it should '
                     'have same length as `current_state`.')
  def maybe_flatten(x):
    return x if maybe_expand or mcmc_util.is_list_like(state) else x[0]
  return [
      maybe_flatten(state_parts),
      maybe_flatten(step_sizes),
      target_log_prob
  ]
```

## test-8645 — OrderedBidictBase.equals_order_sensitive

- url: https://github.com/jab/bidict/blob/1a1ba9758651aed9c4f58384eff006d2e2ad6835/bidict/_orderedbase.py#L288-L296
- repo: jab/bidict

```python
def equals_order_sensitive(self, other):
        



        # Same short-circuit as BidictBase.__eq__. Factoring out not worth function call overhead.
        if not isinstance(other, Mapping) or len(self) != len(other):
            return False
        return all(i == j for (i, j) in izip(iteritems(self), iteritems(other)))
```

## test-1888 — main

- url: https://github.com/deepmipt/DeepPavlov/blob/f3e4a69a3764d25d2f5bad4f1f1aebc872b00f9c/deeppavlov/agents/ecommerce_agent/ecommerce_agent.py#L189-L196
- repo: deepmipt/DeepPavlov

```python
def main():
    

    args = parser.parse_args()
    run_ms_bot_framework_server(agent_generator=make_agent,
                                app_id=args.ms_id,
                                app_secret=args.ms_secret,
                                stateful=True)
```

## test-11745 — write_polfils

- url: https://github.com/UCBerkeleySETI/blimpy/blob/b8822d3e3e911944370d84371a91fa0c29e9772e/blimpy/calib_utils/stokescal.py#L301-L312
- repo: UCBerkeleySETI/blimpy

```python
def write_polfils(str, str_I, **kwargs):
    


    lin,circ=fracpols(str, **kwargs)
    obs = Waterfall(str_I, max_load=150)

    obs.data = lin
    obs.write_to_fil(str[:-15]+'.linpol.fil')   #assuming file is named *.cross_pols.fil

    obs.data = circ
    obs.write_to_fil(str[:-15]+'.circpol.fil')
```

## test-5769 — DotBackend.generate

- url: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/graph.py#L73-L131
- repo: PyCQA/pylint

```python
def generate(self, outputfile=None, dotfile=None, mapfile=None):
        








        import subprocess  # introduced in py 2.4

        name = self.graphname
        if not dotfile:
            # if 'outputfile' is a dot file use it as 'dotfile'
            if outputfile and outputfile.endswith(".dot"):
                dotfile = outputfile
            else:
                dotfile = "%s.dot" % name
        if outputfile is not None:
            storedir, _, target = target_info_from_filename(outputfile)
            if target != "dot":
                pdot, dot_sourcepath = tempfile.mkstemp(".dot", name)
                os.close(pdot)
            else:
                dot_sourcepath = osp.join(storedir, dotfile)
        else:
            target = "png"
            pdot, dot_sourcepath = tempfile.mkstemp(".dot", name)
            ppng, outputfile = tempfile.mkstemp(".png", name)
            os.close(pdot)
            os.close(ppng)
        pdot = codecs.open(dot_sourcepath, "w", encoding="utf8")
        pdot.write(self.source)
        pdot.close()
        if target != "dot":
            use_shell = sys.platform == "win32"
            if mapfile:
                subprocess.call(
                    [
                        self.renderer,
                        "-Tcmapx",
                        "-o",
                        mapfile,
                        "-T",
                        target,
                        dot_sourcepath,
                        "-o",
                        outputfile,
                    ],
                    shell=use_shell,
                )
            else:
                subprocess.call(
                    [self.renderer, "-T", target, dot_sourcepath, "-o", outputfile],
                    shell=use_shell,
                )
            os.unlink(dot_sourcepath)
        return outputfile
```

## test-17711 — Environment.register_defaults

- url: https://github.com/gears/gears/blob/5729c2525a8c04c185e998bd9a86233708972921/gears/environment.py#L307-L311
- repo: gears/gears

```python
def register_defaults(self):
        
        self.mimetypes.register_defaults()
        self.preprocessors.register_defaults()
        self.postprocessors.register_defaults()
```

## test-18910 — ForceNumeric.str_to_num

- url: https://github.com/bheinzerling/descriptors/blob/04fff864649fba9bd6a2d8f8b649cf30994e0e46/descriptors/handmade.py#L167-L176
- repo: bheinzerling/descriptors

```python
def str_to_num(str_value):
        



        str_value = str(str_value)
        try:
            return int(str_value)
        except ValueError:
            return float(str_value)
```

## test-2681 — create_venv_with_package

- url: https://github.com/Azure/azure-sdk-for-python/blob/d7306fde32f60a293a7567678692bdad31e4b667/azure-sdk-tools/packaging_tools/venvtools.py#L30-L46
- repo: Azure/azure-sdk-for-python

```python
def create_venv_with_package(packages):
    



    with tempfile.TemporaryDirectory() as tempdir:
        myenv = create(tempdir, with_pip=True)
        pip_call = [
            myenv.env_exe,
            "-m",
            "pip",
            "install",
        ]
        subprocess.check_call(pip_call + ['-U', 'pip'])
        if packages:
            subprocess.check_call(pip_call + packages)
        yield myenv
```

## test-13435 — map_arguments_to_objects

- url: https://github.com/Patreon/multiget-cache-py/blob/824ec4809c97cc7e0035810bd9fefd1262de3318/multiget_cache/function_tools.py#L105-L124
- repo: Patreon/multiget-cache-py

```python
def map_arguments_to_objects(kwargs, objects, object_key, object_tuple_key, argument_key, result_value, default_result):
    











    # Map each object to the set of desired result data using a key
    # that corresponds to the parameters for ordering purposes
    map_ = map_objects_to_result(objects, object_key, object_tuple_key, result_value, default_result)
    element_count = get_request_count(kwargs)
    # Using the map we just made, return the objects in the same order
    # they were primed using the object and argument keys to match against
    return [map_[get_argument_key(kwargs, argument_key, index)] for index in range(0, element_count)]
```

## test-1904 — squad_v2_exact_match

- url: https://github.com/deepmipt/DeepPavlov/blob/f3e4a69a3764d25d2f5bad4f1f1aebc872b00f9c/deeppavlov/metrics/squad_metrics.py#L24-L40
- repo: deepmipt/DeepPavlov

```python
def squad_v2_exact_match(y_true: List[List[str]], y_predicted: List[str]) -> float:
    












    EM_total = sum(normalize_answer(prediction) in map(normalize_answer, ground_truth)
                   for ground_truth, prediction in zip(y_true, y_predicted))
    return 100 * EM_total / len(y_true) if len(y_true) > 0 else 0
```

## test-20881 — RichIPythonWidget._insert_svg

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/IPython/frontend/qt/console/rich_ipython_widget.py#L301-L313
- repo: cloud9ers/gurumate

```python
def _insert_svg(self, cursor, svg):
        

        try:
            image = svg_to_image(svg)
        except ValueError:
            self._insert_plain_text(cursor, 'Received invalid SVG data.')
        else:
            format = self._add_image(image)
            self._name_to_svg_map[format.name()] = svg
            cursor.insertBlock()
            cursor.insertImage(format)
            cursor.insertBlock()
```

## test-14042 — SlackClientProtocol.make_message

- url: https://github.com/djangobot/djangobot/blob/0ec951891812ea4114c27a08c790f63d0f0fd254/djangobot/client.py#L49-L66
- repo: djangobot/djangobot

```python
def make_message(self, text, channel):
        






        try:
            channel_id = self.slack.channel_from_name(channel)['id']
        except ValueError:
            channel_id = channel
        return pack({
            'text': text,
            'type': 'message',
            'channel': channel_id,
            'id': self.message_id,
        })
```

## test-4156 — DAGCircuit.get_2q_nodes

- url: https://github.com/Qiskit/qiskit-terra/blob/d4f58d903bc96341b816f7c35df936d6421267d1/qiskit/dagcircuit/dagcircuit.py#L1028-L1040
- repo: Qiskit/qiskit-terra

```python
def get_2q_nodes(self):
        
        warnings.warn('The method get_2q_nodes() is being replaced by twoQ_gates()',
                      'Returning a list of data_dicts is also deprecated, twoQ_gates() '
                      'returns a list of DAGNodes.',
                      DeprecationWarning, 2)

        two_q_nodes = []
        for node in self._multi_graph.nodes():
            if node.type == 'op' and len(node.qargs) == 2:
                two_q_nodes.append(node.data_dict)

        return two_q_nodes
```

## test-16111 — handle_requires

- url: https://github.com/AkihikoITOH/capybara/blob/e86c2173ea386654f4ae061148e8fbe3f25e715c/capybara/virtualenv/lib/python2.7/site-packages/wheel/metadata.py#L50-L82
- repo: AkihikoITOH/capybara

```python
def handle_requires(metadata, pkg_info, key):
    


    may_requires = defaultdict(list)
    for value in pkg_info.get_all(key):
        extra_match = EXTRA_RE.search(value)
        if extra_match:
            groupdict = extra_match.groupdict()
            condition = groupdict['condition']
            extra = groupdict['extra']
            package = groupdict['package']
            if condition.endswith(' and '):
                condition = condition[:-5]
        else:
            condition, extra = None, None
            package = value
        key = MayRequiresKey(condition, extra)
        may_requires[key].append(package)

    if may_requires:
        metadata['run_requires'] = []
        for key, value in may_requires.items():
            may_requirement = {'requires':value}
            if key.extra:
                may_requirement['extra'] = key.extra
            if key.condition:
                may_requirement['environment'] = key.condition
            metadata['run_requires'].append(may_requirement)

        if not 'extras' in metadata:
            metadata['extras'] = []
        metadata['extras'].extend([key.extra for key in may_requires.keys() if key.extra])
```

## test-5588 — PyLinter.set_current_module

- url: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/lint.py#L1158-L1170
- repo: PyCQA/pylint

```python
def set_current_module(self, modname, filepath=None):
        


        if not modname and filepath is None:
            return
        self.reporter.on_set_current_module(modname, filepath)
        self.current_name = modname
        self.current_file = filepath or modname
        self.stats["by_module"][modname] = {}
        self.stats["by_module"][modname]["statement"] = 0
        for msg_cat in MSG_TYPES.values():
            self.stats["by_module"][modname][msg_cat] = 0
```

## test-18903 — Graph._get_axis_mode

- url: https://github.com/andrefsp/pyflot/blob/f2dde10709aeed39074fcce8172184b5cd8bfd66/flot/__init__.py#L266-L270
- repo: andrefsp/pyflot

```python
def _get_axis_mode(self, axis):
        
        if all([isinstance(getattr(s, axis), TimeVariable) for s in self._series]):
            return 'time'
        return None
```

## test-3801 — is_diagonal_matrix

- url: https://github.com/Qiskit/qiskit-terra/blob/d4f58d903bc96341b816f7c35df936d6421267d1/qiskit/quantum_info/operators/predicates.py#L74-L83
- repo: Qiskit/qiskit-terra

```python
def is_diagonal_matrix(mat, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    
    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    return np.allclose(mat, np.diag(np.diagonal(mat)), rtol=rtol, atol=atol)
```

## test-2805 — _convert_etree_element_to_rule

- url: https://github.com/Azure/azure-sdk-for-python/blob/d7306fde32f60a293a7567678692bdad31e4b667/azure-servicebus/azure/servicebus/control_client/_serialization.py#L121-L166
- repo: Azure/azure-sdk-for-python

```python
def _convert_etree_element_to_rule(entry_element):
    

















    rule = Rule()

    rule_element = entry_element.find('./atom:content/sb:RuleDescription', _etree_sb_feed_namespaces)
    if rule_element is not None:
        filter_element = rule_element.find('./sb:Filter', _etree_sb_feed_namespaces)
        if filter_element is not None:
            rule.filter_type = filter_element.attrib.get(
                _make_etree_ns_attr_name(_etree_sb_feed_namespaces['i'], 'type'), None)
            sql_exp_element = filter_element.find('./sb:SqlExpression', _etree_sb_feed_namespaces)
            if sql_exp_element is not None:
                rule.filter_expression = sql_exp_element.text

        action_element = rule_element.find('./sb:Action', _etree_sb_feed_namespaces)
        if action_element is not None:
            rule.action_type = action_element.attrib.get(
                _make_etree_ns_attr_name(_etree_sb_feed_namespaces['i'], 'type'), None)
            sql_exp_element = action_element.find('./sb:SqlExpression', _etree_sb_feed_namespaces)
            if sql_exp_element is not None:
                rule.action_expression = sql_exp_element.text


    # extract id, updated and name value from feed entry and set them of rule.
    for name, value in _ETreeXmlToObject.get_entry_properties_from_element(
            entry_element, True, '/rules').items():
        setattr(rule, name, value)

    return rule
```

## test-13322 — _let_to_py_ast

- url: https://github.com/chrisrink10/basilisp/blob/3d82670ee218ec64eb066289c82766d14d18cc92/src/basilisp/lang/compiler/generator.py#L1392-L1426
- repo: chrisrink10/basilisp

```python
def _let_to_py_ast(ctx: GeneratorContext, node: Let) -> GeneratedPyAST:
    
    assert node.op == NodeOp.LET

    with ctx.new_symbol_table("let"):
        let_body_ast: List[ast.AST] = []
        for binding in node.bindings:
            init_node = binding.init
            assert init_node is not None
            init_ast = gen_py_ast(ctx, init_node)
            binding_name = genname(munge(binding.name))
            let_body_ast.extend(init_ast.dependencies)
            let_body_ast.append(
                ast.Assign(
                    targets=[ast.Name(id=binding_name, ctx=ast.Store())],
                    value=init_ast.node,
                )
            )
            ctx.symbol_table.new_symbol(
                sym.symbol(binding.name), binding_name, LocalType.LET
            )

        let_result_name = genname("let_result")
        body_ast = _synthetic_do_to_py_ast(ctx, node.body)
        let_body_ast.extend(map(statementize, body_ast.dependencies))
        let_body_ast.append(
            ast.Assign(
                targets=[ast.Name(id=let_result_name, ctx=ast.Store())],
                value=body_ast.node,
            )
        )

        return GeneratedPyAST(
            node=ast.Name(id=let_result_name, ctx=ast.Load()), dependencies=let_body_ast
        )
```

## test-16500 — Rule.execute_actions

- url: https://github.com/ssherar/hook/blob/54160df554d8b2ed65d762168e5808487e873ed9/hook/model.py#L57-L63
- repo: ssherar/hook

```python
def execute_actions(self, cwd):
        
        self._execute_globals(cwd)
        for action in self.actions:
            logger.info("executing {}".format(action))
            p = subprocess.Popen(action, shell=True, cwd=cwd)
            p.wait()
```

## test-22031 — Rscript.run_command_under_r_root

- url: https://github.com/chairco/pyRscript/blob/e952f450a873de52baa4fe80ed901f0cf990c0b7/pyRscript/pyRscript.py#L88-L98
- repo: chairco/pyRscript

```python
def run_command_under_r_root(self, cmd, catched=True):
        


        RPATH = self.path
        with self.cd(newdir=RPATH):
            if catched:
                process = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            else:
                process = sp.run(cmd)
            return process
```

## test-16973 — plexp_cdf

- url: https://github.com/keflavich/plfit/blob/7dafa6302b427ba8c89651148e3e9d29add436c3/plfit/plfit.py#L796-L815
- repo: keflavich/plfit

```python
def plexp_cdf(x,xmin=1,alpha=2.5, pl_only=False, exp_only=False):
    





    x = np.array(x)
    C = 1/(-xmin/(1 - alpha) - xmin/alpha + exp(alpha)*xmin/alpha)
    Ppl = lambda X: 1+C*(xmin/(1-alpha)*(X/xmin)**(1-alpha))
    Pexp = lambda X: C*xmin/alpha*exp(alpha)-C*(xmin/alpha)*exp(-alpha*(X/xmin-1))

    if exp_only:
        return Pexp(x)
    elif pl_only:
        return Ppl(x)

    d=Ppl(x)
    d[x<xmin]=Pexp(x)[x<xmin]
    return d
```

## test-6085 — dump_privatekey

- url: https://github.com/pyca/pyopenssl/blob/1fbe064c50fd030948141d7d630673761525b0d0/src/OpenSSL/crypto.py#L1851-L1907
- repo: pyca/pyopenssl

```python
def dump_privatekey(type, pkey, cipher=None, passphrase=None):
    














    bio = _new_mem_buf()

    if not isinstance(pkey, PKey):
        raise TypeError("pkey must be a PKey")

    if cipher is not None:
        if passphrase is None:
            raise TypeError(
                "if a value is given for cipher "
                "one must also be given for passphrase")
        cipher_obj = _lib.EVP_get_cipherbyname(_byte_string(cipher))
        if cipher_obj == _ffi.NULL:
            raise ValueError("Invalid cipher name")
    else:
        cipher_obj = _ffi.NULL

    helper = _PassphraseHelper(type, passphrase)
    if type == FILETYPE_PEM:
        result_code = _lib.PEM_write_bio_PrivateKey(
            bio, pkey._pkey, cipher_obj, _ffi.NULL, 0,
            helper.callback, helper.callback_args)
        helper.raise_if_problem()
    elif type == FILETYPE_ASN1:
        result_code = _lib.i2d_PrivateKey_bio(bio, pkey._pkey)
    elif type == FILETYPE_TEXT:
        if _lib.EVP_PKEY_id(pkey._pkey) != _lib.EVP_PKEY_RSA:
            raise TypeError("Only RSA keys are supported for FILETYPE_TEXT")

        rsa = _ffi.gc(
            _lib.EVP_PKEY_get1_RSA(pkey._pkey),
            _lib.RSA_free
        )
        result_code = _lib.RSA_print(bio, rsa, 0)
    else:
        raise ValueError(
            "type argument must be FILETYPE_PEM, FILETYPE_ASN1, or "
            "FILETYPE_TEXT")

    _openssl_assert(result_code != 0)

    return _bio_to_string(bio)
```

## test-154 — GcfHook.delete_function

- url: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_function_hook.py#L161-L172
- repo: apache/airflow

```python
def delete_function(self, name):
        






        response = self.get_conn().projects().locations().functions().delete(
            name=name).execute(num_retries=self.num_retries)
        operation_name = response["name"]
        self._wait_for_operation_to_complete(operation_name=operation_name)
```

## test-2075 — H2OFrame.as_data_frame

- url: https://github.com/h2oai/h2o-3/blob/dd62aaa1e7f680a8b16ee14bc66b0fb5195c2ad8/h2o-py/h2o/frame.py#L1286-L1305
- repo: h2oai/h2o-3

```python
def as_data_frame(self, use_pandas=True, header=True):
        









 
        if can_use_pandas() and use_pandas:
            import pandas
            return pandas.read_csv(StringIO(self.get_frame_data()), low_memory=False, skip_blank_lines=False)
        from h2o.utils.csv.readers import reader
        frame = [row for row in reader(StringIO(self.get_frame_data()))]
        if not header:
            frame.pop(0)
        return frame
```

## test-21875 — IOPubHandler.stop_hb

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/IPython/frontend/html/notebook/handlers.py#L530-L536
- repo: cloud9ers/gurumate

```python
def stop_hb(self):
        
        if self._beating:
            self._beating = False
            self._hb_periodic_callback.stop()
            if not self.hb_stream.closed():
                self.hb_stream.on_recv(None)
```

## test-19914 — SharedLibWrapper.__openlib

- url: https://github.com/sundarnagarajan/cffi_utils/blob/1d5ab2d2fcb962372228033106bc23f1d73d31fa/cffi_utils/sowrapper.py#L104-L140
- repo: sundarnagarajan/cffi_utils

```python
def __openlib(self):
        


        if self.__getattribute__('_libloaded'):
            return
        libpath_list = self.__get_libres()
        for p in libpath_list:
            try:
                libres = resource_filename(self._module_name, p)
                self.lib = self.ffi.dlopen(libres)
                return
            except:
                continue
        # Try self._libpath if nothing in libpath_list worked - will work
        # only if self._module_name is set
        try:
            libres = resource_filename(self._module_name, self._libpath)
            self.lib = self.ffi.dlopen(libres)
        except:
            # If self._module_name is in sys.modules, try self._libpath
            # in the same dir as sys.modules[self._module_name].__file__
            # This is allows get_lib_ffi_shared to work in REPL
            try:
                # We set _libloaded to indicate all options have been tried
                self._libloaded = True

                libdir = ''
                if self._module_name is not None:
                    mod = sys.modules.get(self._module_name, None)
                    if mod is not None:
                        libdir = os.path.dirname(mod.__file__) or os.getcwd()

                libres = os.path.join(libdir, self._libpath)
                self.lib = self.ffi.dlopen(libres)
            except:
                return None
```

## test-15278 — Status.merge

- url: https://github.com/drhagen/parsita/blob/d97414a05541f48231381f607d1d2e6b50781d39/parsita/state.py#L298-L330
- repo: drhagen/parsita

```python
def merge(self, status: 'Status[Input, Output]') -> 'Status[Input, Output]':
        












        if status is None or status.farthest is None:
            # No new message; simply return unchanged
            pass
        elif self.farthest is None:
            # No current message to compare to; use the message from status
            self.farthest = status.farthest
            self.expected = status.expected
        elif status.farthest.position < self.farthest.position:
            # New message is not farther; keep current message
            pass
        elif status.farthest.position > self.farthest.position:
            # New message is farther than current message; replace with new message
            self.farthest = status.farthest
            self.expected = status.expected
        else:
            # New message and current message are equally far; merge messages
            self.expected = status.expected + self.expected

        return self
```

## test-7423 — Client.guild_count

- url: https://github.com/DiscordBotList/DBL-Python-Library/blob/c1461ae0acc644cdeedef8fd6b5e36f76d81c1aa/dbl/client.py#L76-L81
- repo: DiscordBotList/DBL-Python-Library

```python
def guild_count(self):
        
        try:
            return len(self.bot.guilds)
        except AttributeError:
            return len(self.bot.servers)
```

## test-17937 — resources.delete_dataset

- url: https://github.com/neurodata/ndio/blob/792dd5816bc770b05a3db2f4327da42ff6253531/ndio/remote/resources.py#L389-L405
- repo: neurodata/ndio

```python
def delete_dataset(self, name):
        






        url = self.url() + "/resource/dataset/{}".format(name)
        req = self.remote_utils.delete_url(url)

        if req.status_code is not 204:
            raise RemoteDataUploadError('Could not delete {}'.format(req.text))
        if req.content == "" or req.content == b'':
            return True
        else:
            return False
```

## test-18303 — USGSDownload.connect_earthexplorer

- url: https://github.com/lucaslamounier/USGSDownload/blob/0969483ea9f9648aa17b099f36d2e1010488b2a4/usgsdownload/usgs.py#L193-L215
- repo: lucaslamounier/USGSDownload

```python
def connect_earthexplorer(self):
        
        logger.info("Establishing connection to Earthexplorer")
        print("\n Establishing connection to Earthexplorer")
        try:
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
            urllib.request.install_opener(opener)
            params = urllib.parse.urlencode(dict(username=self.user, password=self.password))
            params = params.encode('utf-8')
            f = opener.open("https://ers.cr.usgs.gov/login", params)
            data = f.read().decode('utf-8')
            f.close()
            if data.find(
                    'You must sign in as a registered user to download data or place orders for USGS EROS products') > 0:
                print("\n Authentification failed")
                logger.error("Authentification failed")
                raise AutenticationUSGSFailed('Authentification USGS failed')
            print('User %s connected with USGS' % self.user)
            logger.debug('User %s connected with USGS' % self.user)
            return
        except Exception as e:
            print('\nError when trying to connect USGS: %s' % e)
            raise logger.error('Error when trying to connect USGS: %s' % e)
```

## test-7164 — CryptographyEngine.mac

- url: https://github.com/OpenKMIP/PyKMIP/blob/b51c5b044bd05f8c85a1d65d13a583a4d8fc1b0e/kmip/services/server/crypto/engine.py#L222-L288
- repo: OpenKMIP/PyKMIP

```python
def mac(self, algorithm, key, data):
        
























        mac_data = None

        if algorithm in self._hash_algorithms.keys():
            self.logger.info(
                "Generating a hash-based message authentication code using "
                "{0}".format(algorithm.name)
            )
            hash_algorithm = self._hash_algorithms.get(algorithm)
            try:
                h = hmac.HMAC(key, hash_algorithm(), backend=default_backend())
                h.update(data)
                mac_data = h.finalize()
            except Exception as e:
                self.logger.exception(e)
                raise exceptions.CryptographicFailure(
                    "An error occurred while computing an HMAC. "
                    "See the server log for more information."
                )
        elif algorithm in self._symmetric_key_algorithms.keys():
            self.logger.info(
                "Generating a cipher-based message authentication code using "
                "{0}".format(algorithm.name)
            )
            cipher_algorithm = self._symmetric_key_algorithms.get(algorithm)
            try:
                # ARC4 and IDEA algorithms will raise exception as CMAC
                # requires block ciphers
                c = cmac.CMAC(cipher_algorithm(key), backend=default_backend())
                c.update(data)
                mac_data = c.finalize()
            except Exception as e:
                raise exceptions.CryptographicFailure(
                    "An error occurred while computing a CMAC. "
                    "See the server log for more information."
                )
        else:
            raise exceptions.InvalidField(
                "The cryptographic algorithm ({0}) is not a supported "
                "for a MAC operation.".format(algorithm)
            )
        return mac_data
```

## test-12128 — TableDescriptor.prepare_extensible

- url: https://github.com/openergy/oplus/blob/f095868d1990c1d126e906ada6acbab26348b3d3/oplus/epm/table_descriptor.py#L49-L91
- repo: openergy/oplus

```python
def prepare_extensible(self):
        


        # see if extensible and store cycle len
        for k in self._tags:
            if "extensible" in k:
                cycle_len = int(k.split(":")[1])
                break
        else:
            # not extensible
            return

        # find cycle start and prepare patterns
        cycle_start = None
        cycle_patterns = []
        for i, field_descriptor in enumerate(self._field_descriptors):
            # quit if finished
            if (cycle_start is not None) and (i >= (cycle_start + cycle_len)):
                break

            # set cycle start if not set yet
            if (cycle_start is None) and ("begin-extensible" in field_descriptor.tags):
                cycle_start = i

            # leave if cycle start not reached yet
            if cycle_start is None:
                continue

            # store pattern
            cycle_patterns.append(field_descriptor.ref.replace("1", r"(\d+)"))
        else:
            raise RuntimeError("cycle start not found")

        # detach unnecessary field descriptors
        self._field_descriptors = self._field_descriptors[:cycle_start + cycle_len]

        # store cycle info
        self.extensible_info = (cycle_start, cycle_len, tuple(cycle_patterns))

        # set field descriptor cycle_start index (for error messages while serialization)
        for i, fd in enumerate(self._field_descriptors[cycle_start:]):
            fd.set_extensible_info(cycle_start, cycle_len, cycle_patterns[i])
```

## test-8024 — PostgresContentsManager.purge_db

- url: https://github.com/quantopian/pgcontents/blob/ed36268b7917332d16868208e1e565742a8753e1/pgcontents/pgmanager.py#L107-L112
- repo: quantopian/pgcontents

```python
def purge_db(self):
        


        with self.engine.begin() as db:
            purge_user(db, self.user_id)
```

## test-21287 — _opcode_set

- url: https://github.com/tnkteja/myhelp/blob/fb3a4809d448ad14d5b2e6ddf2e7e89ad52b71cb/virtualEnvironment/lib/python2.7/site-packages/coverage/parser.py#L282-L290
- repo: tnkteja/myhelp

```python
def _opcode_set(*names):
    
    s = set()
    for name in names:
        try:
            s.add(_opcode(name))
        except KeyError:
            pass
    return s
```

## test-4937 — BugzillaClient.bugs

- url: https://github.com/chaoss/grimoirelab-perceval/blob/41c908605e88b7ebc3a536c643fa0f212eaf9e0e/perceval/backends/core/bugzilla.py#L515-L528
- repo: chaoss/grimoirelab-perceval

```python
def bugs(self, *bug_ids):
        



        params = {
            self.PBUG_ID: bug_ids,
            self.PCTYPE: self.CTYPE_XML,
            self.PEXCLUDE_FIELD: 'attachmentdata'
        }

        response = self.call(self.CGI_BUG, params)

        return response
```

## test-8432 — MgmtExtensions.update

- url: https://github.com/ibm-watson-iot/iot-python/blob/195f05adce3fba4ec997017e41e02ebd85c0c4cc/src/wiotp/sdk/api/mgmt/extensions.py#L78-L90
- repo: ibm-watson-iot/iot-python

```python
def update(self, bundleId, dmeData):
        




        url = "api/v0002/mgmt/custom/bundle/%s" % (bundleId)
        r = self._apiClient.put(url, dmeData)

        if r.status_code == 200:
            return r.json()
        else:
            raise ApiException(r)
```

## test-18167 — Resource._post

- url: https://github.com/ambitioninc/python-logentries-api/blob/77ff1a7a2995d7ea2725b74e34c0f880f4ee23bc/logentries_api/base.py#L49-L88
- repo: ambitioninc/python-logentries-api

```python
def _post(self, request, uri, params=None):
        




















        request_data = {
            'acl': self.account_key,
            'account': self.account_key,
            'request': request,
        }

        request_data.update(params or {})

        response = requests.post(
            url='https://api.logentries.com/v2/{}'.format(uri),
            headers=self.headers,
            data=json.dumps(request_data)
        )

        if not response.ok:
            raise ServerException(
                '{}: {}'.format(response.status_code, response.text))
        return response.json()
```

## test-752 — CloudSpannerHook.execute_dml

- url: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/gcp_spanner_hook.py#L328-L344
- repo: apache/airflow

```python
def execute_dml(self, instance_id, database_id, queries, project_id=None):
        












        self._get_client(project_id=project_id).instance(instance_id=instance_id).\
            database(database_id=database_id).run_in_transaction(
            lambda transaction: self._execute_sql_in_transaction(transaction, queries))
```

## test-21935 — TaskScheduler.save_unmet

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/IPython/parallel/controller/scheduler.py#L533-L541
- repo: cloud9ers/gurumate

```python
def save_unmet(self, job):
        
        msg_id = job.msg_id
        self.depending[msg_id] = job
        # track the ids in follow or after, but not those already finished
        for dep_id in job.after.union(job.follow).difference(self.all_done):
            if dep_id not in self.graph:
                self.graph[dep_id] = set()
            self.graph[dep_id].add(msg_id)
```

## test-2201 — H2OAutoEncoderGridSearch.anomaly

- url: https://github.com/h2oai/h2o-3/blob/dd62aaa1e7f680a8b16ee14bc66b0fb5195c2ad8/h2o-py/h2o/grid/metrics.py#L11-L20
- repo: h2oai/h2o-3

```python
def anomaly(self, test_data, per_feature=False):
        







        return {model.model_id: model.anomaly(test_data, per_feature) for model in self.models}
```

## test-21599 — ConsoleWidget.change_font_size

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/IPython/frontend/qt/console/console_widget.py#L746-L752
- repo: cloud9ers/gurumate

```python
def change_font_size(self, delta):
        

        font = self.font
        size = max(font.pointSize() + delta, 1) # minimum 1 point
        font.setPointSize(size)
        self._set_font(font)
```

## test-21982 — repr_type

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/IPython/utils/traitlets.py#L108-L117
- repo: cloud9ers/gurumate

```python
def repr_type(obj):
    


    the_type = type(obj)
    if (not py3compat.PY3) and the_type is InstanceType:
        # Old-style class.
        the_type = obj.__class__
    msg = '%r %r' % (obj, the_type)
    return msg
```

## test-245 — template_field_role

- url: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/docs/exts/docroles.py#L58-L88
- repo: apache/airflow

```python
def template_field_role(app, typ, rawtext, text, lineno, inliner, options={}, content=[]):
    













    text = utils.unescape(text)

    try:
        template_fields = get_template_field(app.env, text)
    except RoleException as e:
        msg = inliner.reporter.error("invalid class name %s \n%s" % (text, e, ), line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    node = nodes.inline(rawtext=rawtext)
    for i, field in enumerate(template_fields):
        if i != 0:
            node += nodes.Text(", ")
        node += nodes.literal(field, "", nodes.Text(field))

    return [node], []
```

## test-334 — DagBag.collect_dags

- url: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/models/dagbag.py#L341-L396
- repo: apache/airflow

```python
def collect_dags(
            self,
            dag_folder=None,
            only_if_updated=True,
            include_examples=configuration.conf.getboolean('core', 'LOAD_EXAMPLES'),
            safe_mode=configuration.conf.getboolean('core', 'DAG_DISCOVERY_SAFE_MODE')):
        











        start_dttm = timezone.utcnow()
        dag_folder = dag_folder or self.dag_folder

        # Used to store stats around DagBag processing
        stats = []
        FileLoadStat = namedtuple(
            'FileLoadStat', "file duration dag_num task_num dags")

        dag_folder = correct_maybe_zipped(dag_folder)

        for filepath in list_py_file_paths(dag_folder, safe_mode=safe_mode,
                                           include_examples=include_examples):
            try:
                ts = timezone.utcnow()
                found_dags = self.process_file(
                    filepath, only_if_updated=only_if_updated,
                    safe_mode=safe_mode)

                td = timezone.utcnow() - ts
                td = td.total_seconds() + (
                    float(td.microseconds) / 1000000)
                stats.append(FileLoadStat(
                    filepath.replace(dag_folder, ''),
                    td,
                    len(found_dags),
                    sum([len(dag.tasks) for dag in found_dags]),
                    str([dag.dag_id for dag in found_dags]),
                ))
            except Exception as e:
                self.log.exception(e)
        Stats.gauge(
            'collect_dags', (timezone.utcnow() - start_dttm).total_seconds(), 1)
        Stats.gauge(
            'dagbag_size', len(self.dags), 1)
        Stats.gauge(
            'dagbag_import_errors', len(self.import_errors), 1)
        self.dagbag_stats = sorted(
            stats, key=lambda x: x.duration, reverse=True)
```

## test-18395 — MetricMarkdown.getFieldsColumnLengths

- url: https://github.com/boundary/pulse-api-cli/blob/b01ca65b442eed19faac309c9d62bbc3cb2c098f/boundary/metric_markdown.py#L105-L114
- repo: boundary/pulse-api-cli

```python
def getFieldsColumnLengths(self):
        


        nameLen = 0
        descLen = 0
        for f in self.fields:
            nameLen = max(nameLen, len(f['title']))
            descLen = max(descLen, len(f['description']))
        return (nameLen, descLen)
```

## test-19354 — RestfulWorker.delete

- url: https://github.com/hivetech/dna/blob/50ad00031be29765b2576fa407d35a36e0608de9/python/dna/apy/worker.py#L117-L135
- repo: hivetech/dna

```python
def delete(self, worker_id):
        
        code = 200

        if worker_id in self.jobs:
            # NOTE pop it if done ?
            self.jobs[worker_id]['worker'].revoke(terminate=True)
            report = {
                'id': worker_id,
                'revoked': True
                # FIXME Unable to serialize self.jobs[worker_id]
                # 'session': self.jobs.pop(worker_id)
            }
            self.jobs.pop(worker_id)
        else:
            report = {'error': 'job {} unknown'.format(worker_id)}
            code = 404

        return flask.jsonify(report), code
```

## test-8775 — Session.save

- url: https://github.com/josiahcarlson/rom/blob/8b5607a856341df85df33422accc30ba9294dbdb/rom/util.py#L520-L554
- repo: josiahcarlson/rom

```python
def save(self, *objects, **kwargs):
        












        from rom import Model
        full = kwargs.get('full')
        all = kwargs.get('all')
        force = kwargs.get('force')
        changes = 0
        items = deque()
        items.extend(objects)
        while items:
            o = items.popleft()
            if isinstance(o, (list, tuple)):
                items.extendleft(reversed(o))
            elif isinstance(o, Model):
                if not o._deleted and (all or o._modified):
                    changes += o.save(full, force)

            else:
                raise ORMError(
                    "Cannot save an object that is not an instance of a Model (you provided %r)"%(
                        o,))

        return changes
```

## test-7316 — SiteTree.init_tree

- url: https://github.com/idlesign/django-sitetree/blob/61de4608e6e415247c75fe8691027d7c4ed0d1e7/sitetree/sitetreeapp.py#L709-L737
- repo: idlesign/django-sitetree

```python
def init_tree(self, tree_alias, context):
        









        request = context.get('request', None)

        if request is None:
            raise SiteTreeError(
                'Sitetree requires "django.core.context_processors.request" template context processor to be active. '
                'If it is, check that your view pushes request data into the template.')

        if id(request) != id(self.current_request):
            self.init(context)

        # Resolve tree_alias from the context.
        tree_alias = self.resolve_var(tree_alias)
        tree_alias, sitetree_items = self.get_sitetree(tree_alias)

        if not sitetree_items:
            return None, None

        return tree_alias, sitetree_items
```

## test-19541 — GChart.image

- url: https://github.com/appknox/google-chartwrapper/blob/3769aecbef6c83b6cd93ee72ece478ffe433ac57/GChartWrapper/GChart.py#L611-L628
- repo: appknox/google-chartwrapper

```python
def image(self):
        




        try:
            try:
                import Image
            except ImportError:
                from PIL import Image
        except ImportError:
            raise ImportError('You must install PIL to fetch image objects')
        try:
            from cStringIO import StringIO
        except ImportError:
            from StringIO import StringIO
        return Image.open(StringIO(self.urlopen().read()))
```

## test-19229 — Raster.SetGeoTransform

- url: https://github.com/bkg/greenwich/blob/57ec644dadfe43ce0ecf2cfd32a2de71e0c8c141/greenwich/raster.py#L410-L421
- repo: bkg/greenwich

```python
def SetGeoTransform(self, affine):
        






        if isinstance(affine, collections.Sequence):
            affine = AffineTransform(*affine)
        self._affine = affine
        self.ds.SetGeoTransform(affine)
```

## test-2058 — H2OFrame.summary

- url: https://github.com/h2oai/h2o-3/blob/dd62aaa1e7f680a8b16ee14bc66b0fb5195c2ad8/h2o-py/h2o/frame.py#L459-L480
- repo: h2oai/h2o-3

```python
def summary(self, return_data=False):
        






        if not self._has_content():
            print("This H2OFrame is empty and not initialized.")
            return self._ex._cache._data;
        if not self._ex._cache.is_valid(): self._frame()._ex._cache.fill()
        if not return_data:
            if self.nrows == 0:
                print("This H2OFrame is empty.")
            elif H2ODisplay._in_ipy():
                import IPython.display
                IPython.display.display_html(self._ex._cache._tabulate("html", True), raw=True)
            else:
                print(self._ex._cache._tabulate("simple", True))
        else:
            return self._ex._cache._data
```

## test-7482 — _SdcCommon.create_sysdig_capture

- url: https://github.com/draios/python-sdc-client/blob/47f83415842048778939b90944f64386a3bcb205/sdcclient/_common.py#L521-L563
- repo: draios/python-sdc-client

```python
def create_sysdig_capture(self, hostname, capture_name, duration, capture_filter='', folder='/'):
        















        res = self.get_connected_agents()
        if not res[0]:
            return res

        capture_agent = None

        for agent in res[1]:
            if hostname == agent['hostName']:
                capture_agent = agent
                break

        if capture_agent is None:
            return [False, hostname + ' not found']

        data = {
            'agent': capture_agent,
            'name': capture_name,
            'duration': duration,
            'folder': folder,
            'filters': capture_filter,
            'bucketName': '',
            'source': self.product
        }

        res = requests.post(self.url + '/api/sysdig', headers=self.hdrs, data=json.dumps(data), verify=self.ssl_verify)
        return self._request_result(res)
```

## test-520 — SageMakerHook.configure_s3_resources

- url: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/sagemaker_hook.py#L157-L178
- repo: apache/airflow

```python
def configure_s3_resources(self, config):
        






        s3_operations = config.pop('S3Operations', None)

        if s3_operations is not None:
            create_bucket_ops = s3_operations.get('S3CreateBucket', [])
            upload_ops = s3_operations.get('S3Upload', [])
            for op in create_bucket_ops:
                self.s3_hook.create_bucket(bucket_name=op['Bucket'])
            for op in upload_ops:
                if op['Tar']:
                    self.tar_and_s3_upload(op['Path'], op['Key'],
                                           op['Bucket'])
                else:
                    self.s3_hook.load_file(op['Path'], op['Key'],
                                           op['Bucket'])
```

## test-18464 — Scope.difference_update

- url: https://github.com/LionelAuroux/pyrser/blob/f153a97ef2b6bf915a1ed468c0252a9a59b754d5/pyrser/type_system/scope.py#L217-L223
- repo: LionelAuroux/pyrser

```python
def difference_update(self, oset: Scope) -> Scope:
        
        keys = list(self._hsig.keys())
        for k in keys:
            if k in oset:
                del self._hsig[k]
        return self
```

## test-19642 — ProxyingProtocol.dataReceived

- url: https://github.com/lvh/txampext/blob/a7d6cb9f1e9200dba597378cd40eb6a2096d4fd9/txampext/multiplexing.py#L303-L317
- repo: lvh/txampext

```python
def dataReceived(self, data):
        





        log.msg("{} bytes of data received locally".format(len(data)))
        if self.connection is None:
            # we haven't finished connecting yet
            log.msg("Connection not made yet, buffering...")
            self._buffer.write(data)
        else:
            log.msg("Sending data...")
            self._sendData(data)
```

## test-5066 — GitHub.__get_user

- url: https://github.com/chaoss/grimoirelab-perceval/blob/41c908605e88b7ebc3a536c643fa0f212eaf9e0e/perceval/backends/core/github.py#L447-L461
- repo: chaoss/grimoirelab-perceval

```python
def __get_user(self, login):
        

        user = {}

        if not login:
            return user

        user_raw = self.client.user(login)
        user = json.loads(user_raw)
        user_orgs_raw = \
            self.client.user_orgs(login)
        user['organizations'] = json.loads(user_orgs_raw)

        return user
```

## test-6156 — PKCS12.set_privatekey

- url: https://github.com/pyca/pyopenssl/blob/1fbe064c50fd030948141d7d630673761525b0d0/src/OpenSSL/crypto.py#L2384-L2395
- repo: pyca/pyopenssl

```python
def set_privatekey(self, pkey):
        







        if not isinstance(pkey, PKey):
            raise TypeError("pkey must be a PKey instance")
        self._pkey = pkey
```

## test-13473 — _parse_receive

- url: https://github.com/MartinHjelmare/leicacam/blob/1df37bccd34884737d3b5e169fae71dd2f21f1e2/leicacam/cam.py#L141-L161
- repo: MartinHjelmare/leicacam

```python
def _parse_receive(incomming):
    












    debug(b'< ' + incomming)
    # remove terminating null byte
    incomming = incomming.rstrip(b'\x00')
    # split received messages
    # return as list of several messages received
    msgs = incomming.splitlines()
    return [bytes_as_dict(msg) for msg in msgs]
```

## test-19658 — get_terminal_size

- url: https://github.com/MinchinWeb/minchin.text/blob/4d136039561892c3adab49fe81b2f246e5a1507b/minchin/text.py#L352-L374
- repo: MinchinWeb/minchin.text

```python
def get_terminal_size():
    



    try:
        # shutil.get_terminal_size was added to the standard
        # library in Python 3.3
        try:
            from shutil import get_terminal_size as _get_terminal_size  # pylint: disable=no-name-in-module
        except ImportError:
            from backports.shutil_get_terminal_size import get_terminal_size as _get_terminal_size  # pylint: disable=import-error

        sz = _get_terminal_size()
    except ValueError:
        """
        This can result from the 'underlying buffer being detached', which
        occurs during running the unittest on Windows (but not on Linux?)
        """
        terminal_size = namedtuple('Terminal_Size', 'columns lines')
        sz = terminal_size(80, 24)

    return sz
```

## test-7370 — zipkin_span.add_sa_binary_annotation

- url: https://github.com/Yelp/py_zipkin/blob/0944d9a3fb1f1798dbb276694aeed99f2b4283ba/py_zipkin/zipkin.py#L545-L582
- repo: Yelp/py_zipkin

```python
def add_sa_binary_annotation(
        self,
        port=0,
        service_name='unknown',
        host='127.0.0.1',
    ):
        













        if self.kind != Kind.CLIENT:
            # TODO: trying to set a sa binary annotation for a non-client span
            # should result in a logged error
            return

        remote_endpoint = create_endpoint(
            port=port,
            service_name=service_name,
            host=host,
        )
        if not self.logging_context:
            if self.remote_endpoint is not None:
                raise ValueError('SA annotation already set.')
            self.remote_endpoint = remote_endpoint
        else:
            if self.logging_context.remote_endpoint is not None:
                raise ValueError('SA annotation already set.')
            self.logging_context.remote_endpoint = remote_endpoint
```

## test-1078 — DisentangledSequentialVAE.generate

- url: https://github.com/tensorflow/probability/blob/e87fe34111d68c35db0f9eeb4935f1ece9e1a8f5/tensorflow_probability/examples/disentangled_vae.py#L758-L785
- repo: tensorflow/probability

```python
def generate(self, batch_size, length, samples=1, fix_static=False,
               fix_dynamic=False):
    




















    static_sample, _ = self.sample_static_prior(samples, batch_size, fix_static)
    dynamic_sample, _ = self.sample_dynamic_prior(samples, batch_size, length,
                                                  fix_dynamic)
    likelihood = self.decoder((dynamic_sample, static_sample))
    return likelihood
```

## test-20903 — IPCompleter._greedy_changed

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/IPython/core/completer.py#L394-L402
- repo: cloud9ers/gurumate

```python
def _greedy_changed(self, name, old, new):
        
        if new:
            self.splitter.delims = GREEDY_DELIMS
        else:
            self.splitter.delims = DELIMS

        if self.readline:
            self.readline.set_completer_delims(self.splitter.delims)
```

## test-18908 — DashboardRunner.cycle

- url: https://github.com/SketchingDev/Doodle-Dashboard/blob/4d7f4c248875f82a962c275009aac4aa76bd0320/doodledashboard/dashboard.py#L39-L46
- repo: SketchingDev/Doodle-Dashboard

```python
def cycle(self):
        


        messages = self.poll_datafeeds()
        notifications = self.process_notifications(messages)

        self.draw_notifications(notifications)
```

## test-21354 — Process.get_process_gids

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/psutil/_psbsd.py#L208-L211
- repo: cloud9ers/gurumate

```python
def get_process_gids(self):
        
        real, effective, saved = _psutil_bsd.get_process_gids(self.pid)
        return nt_gids(real, effective, saved)
```

## test-5944 — OAuth1RequestValidator.save_request_token

- url: https://github.com/lepture/flask-oauthlib/blob/9e6f152a5bb360e7496210da21561c3e6d41b0e1/flask_oauthlib/provider/oauth1.py#L860-L877
- repo: lepture/flask-oauthlib

```python
def save_request_token(self, token, request):
        














        log.debug('Save request token %r', token)
        self._grantsetter(token, request)
```

## test-11488 — load_scout

- url: https://github.com/Clinical-Genomics/scout/blob/90a551e2e1653a319e654c2405c2866f93d0ebb9/scout/load/all.py#L90-L105
- repo: Clinical-Genomics/scout

```python
def load_scout(adapter, config, ped=None, update=False):
    








    log.info("Check that the panels exists")
    if not check_panels(adapter, config.get('gene_panels', []),
                        config.get('default_gene_panels')):
        raise ConfigError("Some panel(s) does not exist in the database")
    case_obj = adapter.load_case(config, update=update)
    return case_obj
```

## test-734 — BackfillJob._manage_executor_state

- url: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/jobs.py#L1979-L2007
- repo: apache/airflow

```python
def _manage_executor_state(self, running):
        





        executor = self.executor

        for key, state in list(executor.get_event_buffer().items()):
            if key not in running:
                self.log.warning(
                    "%s state %s not in running=%s",
                    key, state, running.values()
                )
                continue

            ti = running[key]
            ti.refresh_from_db()

            self.log.debug("Executor state: %s task %s", state, ti)

            if state == State.FAILED or state == State.SUCCESS:
                if ti.state == State.RUNNING or ti.state == State.QUEUED:
                    msg = ("Executor reports task instance {} finished ({}) "
                           "although the task says its {}. Was the task "
                           "killed externally?".format(ti, state, ti.state))
                    self.log.error(msg)
                    ti.handle_failure(msg)
```

## test-9653 — walkParams

- url: https://github.com/Nic30/hwt/blob/8cbb399e326da3b22c233b98188a9d08dec057e6/hwt/synthesizer/interfaceLevel/interfaceUtils/utils.py#L22-L32
- repo: Nic30/hwt

```python
def walkParams(intf, discovered):
    


    for si in intf._interfaces:
        yield from walkParams(si, discovered)

    for p in intf._params:
        if p not in discovered:
            discovered.add(p)
            yield p
```

## test-14551 — singledispatchmethod

- url: https://github.com/its-rigs/Trolly/blob/483dc94c352df40dc05ead31820b059b2545cf82/trolly/lib.py#L5-L16
- repo: its-rigs/Trolly

```python
def singledispatchmethod(method):
    




    dispatcher = singledispatch(method)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, dispatcher)
    return wrapper
```

## test-3013 — Crunchyroll._create_api

- url: https://github.com/streamlink/streamlink/blob/c8ed1daff14ac03195870238b9b900c1109dd5c1/src/streamlink/plugins/crunchyroll.py#L346-L389
- repo: streamlink/streamlink

```python
def _create_api(self):
        



        if self.options.get("purge_credentials"):
            self.cache.set("session_id", None, 0)
            self.cache.set("auth", None, 0)
            self.cache.set("session_id", None, 0)

        # use the crunchyroll locale as an override, for backwards compatibility
        locale = self.get_option("locale") or self.session.localization.language_code
        api = CrunchyrollAPI(self.cache,
                             self.session,
                             session_id=self.get_option("session_id"),
                             locale=locale)

        if not self.get_option("session_id"):
            self.logger.debug("Creating session with locale: {0}", locale)
            api.start_session()

            if api.auth:
                self.logger.debug("Using saved credentials")
                login = api.authenticate()
                self.logger.info("Successfully logged in as '{0}'",
                                 login["user"]["username"] or login["user"]["email"])
            elif self.options.get("username"):
                try:
                    self.logger.debug("Attempting to login using username and password")
                    api.login(self.options.get("username"),
                              self.options.get("password"))
                    login = api.authenticate()
                    self.logger.info("Logged in as '{0}'",
                                     login["user"]["username"] or login["user"]["email"])

                except CrunchyrollAPIError as err:
                    raise PluginError(u"Authentication error: {0}".format(err.msg))
            else:
                self.logger.warning(
                    "No authentication provided, you won't be able to access "
                    "premium restricted content"
                )

        return api
```

## test-10537 — NextflowInspector._clear_inspect

- url: https://github.com/assemblerflow/flowcraft/blob/fc3f4bddded1efc76006600016dc71a06dd908c0/flowcraft/generator/inspect.py#L456-L475
- repo: assemblerflow/flowcraft

```python
def _clear_inspect(self):
        

        self.trace_info = defaultdict(list)
        self.process_tags = {}
        self.process_stats = {}
        self.samples = []
        self.stored_ids = []
        self.stored_log_ids = []
        self.time_start = None
        self.time_stop = None
        self.execution_command = None
        self.nextflow_version = None
        self.abort_cause = None
        self._c = 0
        # Clean up of tag running status
        for p in self.processes.values():
            p["barrier"] = "W"
            for i in ["submitted", "finished", "failed", "retry"]:
                p[i] = set()
```

## test-20999 — compare_content_type

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/gurumate-2.8.6-py2.7.egg/gurumate/web.py#L57-L68
- repo: cloud9ers/gurumate

```python
def compare_content_type(url, content_type):
    




    try:    
        response = urllib2.urlopen(url)
    except:
        return False

    return response.headers.type == content_type
```

## test-16441 — BSpline._draw_mainlayer

- url: https://github.com/rwl/godot/blob/013687c9e8983d2aa2ceebb8a76c5c4f1e37c90f/godot/component/bspline.py#L124-L147
- repo: rwl/godot

```python
def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        

        if not self.points: return
        gc.save_state()
        try:
            gc.set_fill_color(self.pen.fill_color_)

            gc.set_line_width(self.pen.line_width)
            gc.set_stroke_color(self.pen.color_)

            gc.begin_path()
            start_x, start_y = self.points[0]
            gc.move_to(start_x, start_y)
            for triple in nsplit(self.points[1:], 3):
                x1, y1 = triple[0]
                x2, y2 = triple[1]
                end_x, end_y = triple[2]
                gc.curve_to(x1, y1, x2, y2, end_x, end_y)
                # One point overlap
                gc.move_to(end_x, end_y)
            gc.stroke_path()
        finally:
            gc.restore_state()
```

## test-7080 — PublicKey.validate

- url: https://github.com/OpenKMIP/PyKMIP/blob/b51c5b044bd05f8c85a1d65d13a583a4d8fc1b0e/kmip/pie/objects.py#L806-L845
- repo: OpenKMIP/PyKMIP

```python
def validate(self):
        





        if not isinstance(self.value, bytes):
            raise TypeError("key value must be bytes")
        elif not isinstance(self.cryptographic_algorithm,
                            enums.CryptographicAlgorithm):
            raise TypeError("key algorithm must be a CryptographicAlgorithm "
                            "enumeration")
        elif not isinstance(self.cryptographic_length, six.integer_types):
            raise TypeError("key length must be an integer")
        elif not isinstance(self.key_format_type, enums.KeyFormatType):
            raise TypeError("key format type must be a KeyFormatType "
                            "enumeration")
        elif self.key_format_type not in self._valid_formats:
            raise ValueError("key format type must be one of {0}".format(
                self._valid_formats))

        # TODO (peter-hamilton) Verify that the key bytes match the key format

        mask_count = len(self.cryptographic_usage_masks)
        for i in range(mask_count):
            mask = self.cryptographic_usage_masks[i]
            if not isinstance(mask, enums.CryptographicUsageMask):
                position = "({0} in list)".format(i)
                raise TypeError(
                    "key mask {0} must be a CryptographicUsageMask "
                    "enumeration".format(position))

        name_count = len(self.names)
        for i in range(name_count):
            name = self.names[i]
            if not isinstance(name, six.string_types):
                position = "({0} in list)".format(i)
                raise TypeError("key name {0} must be a string".format(
                    position))
```

## test-21190 — KernelManager.hb_channel

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/IPython/zmq/kernelmanager.py#L987-L994
- repo: cloud9ers/gurumate

```python
def hb_channel(self):
        

        if self._hb_channel is None:
            self._hb_channel = self.hb_channel_class(self.context,
                                                       self.session,
                                                       (self.ip, self.hb_port))
        return self._hb_channel
```

## test-5878 — McCabeMethodChecker.visit_module

- url: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/extensions/mccabe.py#L174-L191
- repo: PyCQA/pylint

```python
def visit_module(self, node):
        

        visitor = PathGraphingAstVisitor()
        for child in node.body:
            visitor.preorder(child, visitor)
        for graph in visitor.graphs.values():
            complexity = graph.complexity()
            node = graph.root
            if hasattr(node, "name"):
                node_name = "'%s'" % node.name
            else:
                node_name = "This '%s'" % node.__class__.__name__.lower()
            if complexity <= self.config.max_complexity:
                continue
            self.add_message(
                "too-complex", node=node, confidence=HIGH, args=(node_name, complexity)
            )
```

## test-7308 — SiteTree.init

- url: https://github.com/idlesign/django-sitetree/blob/61de4608e6e415247c75fe8691027d7c4ed0d1e7/sitetree/sitetreeapp.py#L378-L391
- repo: idlesign/django-sitetree

```python
def init(self, context):
        



        self.cache = Cache()
        self.current_page_context = context
        self.current_request = context.get('request', None) if context else None
        self.current_lang = get_language()

        self._current_app_is_admin = None
        self._current_user_permissions = _UNSET
        self._items_urls = {}  # Resolved urls are cache for a request.
        self._current_items = {}
```

## test-21178 — KernelManager.load_connection_file

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/IPython/zmq/kernelmanager.py#L748-L758
- repo: cloud9ers/gurumate

```python
def load_connection_file(self):
        
        with open(self.connection_file) as f:
            cfg = json.loads(f.read())
        
        self.ip = cfg['ip']
        self.shell_port = cfg['shell_port']
        self.stdin_port = cfg['stdin_port']
        self.iopub_port = cfg['iopub_port']
        self.hb_port = cfg['hb_port']
        self.session.key = str_to_bytes(cfg['key'])
```

## test-11215 — add_incomplete_penetrance

- url: https://github.com/Clinical-Genomics/scout/blob/90a551e2e1653a319e654c2405c2866f93d0ebb9/scout/utils/link.py#L136-L141
- repo: Clinical-Genomics/scout

```python
def add_incomplete_penetrance(genes, alias_genes, hpo_lines):
    
    LOG.info("Add incomplete penetrance info")
    for hgnc_symbol in get_incomplete_penetrance_genes(hpo_lines):
        for hgnc_id in get_correct_ids(hgnc_symbol, alias_genes):
            genes[hgnc_id]['incomplete_penetrance'] = True
```

## test-384 — S3Hook.select_key

- url: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/hooks/S3_hook.py#L233-L277
- repo: apache/airflow

```python
def select_key(self, key, bucket_name=None,
                   expression='SELECT * FROM S3Object',
                   expression_type='SQL',
                   input_serialization=None,
                   output_serialization=None):
        





















        if input_serialization is None:
            input_serialization = {'CSV': {}}
        if output_serialization is None:
            output_serialization = {'CSV': {}}
        if not bucket_name:
            (bucket_name, key) = self.parse_s3_url(key)

        response = self.get_conn().select_object_content(
            Bucket=bucket_name,
            Key=key,
            Expression=expression,
            ExpressionType=expression_type,
            InputSerialization=input_serialization,
            OutputSerialization=output_serialization)

        return ''.join(event['Records']['Payload'].decode('utf-8')
                       for event in response['Payload']
                       if 'Records' in event)
```

## test-6939 — Command.run

- url: https://github.com/funilrys/PyFunceble/blob/cdf69cbde120199171f7158e1c33635753e6e2f5/PyFunceble/helpers.py#L315-L346
- repo: funilrys/PyFunceble

```python
def run(self):
        








        with Popen(self.command, stdout=PIPE, shell=True) as process:
            # We initiate a process and parse the command to it.

            while True:
                # We loop infinitly because we want to get the output
                # until there is none.

                # We get the current line from the process stdout.
                #
                # Note: we use rstrip() because we are paranoid :-)
                current_line = process.stdout.readline().rstrip()

                if not current_line:
                    # The current line is empty or equal to None.

                    # We break the loop.
                    break

                # The line is not empty nor equal to None.

                # We encode and yield the current line
                yield self._decode_output(current_line)
```

## test-6128 — X509.gmtime_adj_notBefore

- url: https://github.com/pyca/pyopenssl/blob/1fbe064c50fd030948141d7d630673761525b0d0/src/OpenSSL/crypto.py#L1310-L1321
- repo: pyca/pyopenssl

```python
def gmtime_adj_notBefore(self, amount):
        





        if not isinstance(amount, int):
            raise TypeError("amount must be an integer")

        notBefore = _lib.X509_get_notBefore(self._x509)
        _lib.X509_gmtime_adj(notBefore, amount)
```

## test-7601 — Bundle.is_confirmed

- url: https://github.com/iotaledger/iota.lib.py/blob/97cdd1e241498446b46157b79b2a1ea2ec6d387a/iota/transaction/base.py#L467-L475
- repo: iotaledger/iota.lib.py

```python
def is_confirmed(self, new_is_confirmed):
        # type: (bool) -> None
        


        self._is_confirmed = new_is_confirmed

        for txn in self:
            txn.is_confirmed = new_is_confirmed
```

## test-15521 — get_model_class_from_string

- url: https://github.com/idlesign/django-etc/blob/dbfc7e9dfc4fdfe69547f71ba4921989f9e97dbe/etc/toolbox.py#L90-L114
- repo: idlesign/django-etc

```python
def get_model_class_from_string(model_path):
    






    try:
        app_name, model_name = model_path.split('.')
    except ValueError:
        raise ImproperlyConfigured('`%s` must have the following format: `app_name.model_name`.' % model_path)

    if apps_get_model is None:
        model = get_model(app_name, model_name)
    else:
        try:
            model = apps_get_model(app_name, model_name)
        except (LookupError, ValueError):
            model = None

    if model is None:
        raise ImproperlyConfigured('`%s` refers to a model `%s` that has not been installed.' % (model_path, model_name))

    return model
```

## test-6118 — X509.from_cryptography

- url: https://github.com/pyca/pyopenssl/blob/1fbe064c50fd030948141d7d630673761525b0d0/src/OpenSSL/crypto.py#L1084-L1100
- repo: pyca/pyopenssl

```python
def from_cryptography(cls, crypto_cert):
        









        if not isinstance(crypto_cert, x509.Certificate):
            raise TypeError("Must be a certificate")

        cert = cls()
        cert._x509 = crypto_cert._x509
        return cert
```

## test-14309 — Eaf.to_textgrid

- url: https://github.com/dopefishh/pympi/blob/79c747cde45b5ba203ed93154d8c123ac9c3ef56/pympi/Elan.py#L1331-L1355
- repo: dopefishh/pympi

```python
def to_textgrid(self, filtin=[], filtex=[], regex=False):
        








        from pympi.Praat import TextGrid
        _, end = self.get_full_time_interval()
        tgout = TextGrid(xmax=end/1000.0)
        func = (lambda x, y: re.match(x, y)) if regex else lambda x, y: x == y
        for tier in self.tiers:
            if (filtin and not any(func(f, tier) for f in filtin)) or\
                    (filtex and any(func(f, tier) for f in filtex)):
                continue
            ctier = tgout.add_tier(tier)
            for intv in self.get_annotation_data_for_tier(tier):
                try:
                    ctier.add_interval(intv[0]/1000.0, intv[1]/1000.0, intv[2])
                except:
                    pass
        return tgout
```

## test-18348 — Compiler._hoist_operands

- url: https://github.com/treycucco/pyebnf/blob/3634ddabbe5d73508bcc20f4a591f86a46634e1d/pyebnf/compiler.py#L273-L287
- repo: treycucco/pyebnf

```python
def _hoist_operands(self, operands, pred):
    




    hopper = list(operands)
    new_operands = []
    while hopper:
      target = hopper.pop(0)
      if pred(target):
        hopper = list(target.operands) + hopper
      else:
        new_operands.append(target)
    return new_operands
```

## test-10514 — Assembly.get_assembly_length

- url: https://github.com/assemblerflow/flowcraft/blob/fc3f4bddded1efc76006600016dc71a06dd908c0/flowcraft/templates/process_viral_assembly.py#L393-L405
- repo: assemblerflow/flowcraft

```python
def get_assembly_length(self):
        








        return sum(
            [vals["length"] for contig_id, vals in self.contigs.items()
             if contig_id not in self.filtered_ids])
```

## test-21549 — canonicalize_path

- url: https://github.com/rackerlabs/timid/blob/b1c6aa159ab380a033740f4aa392cf0d125e0ac6/timid/utils.py#L28-L43
- repo: rackerlabs/timid

```python
def canonicalize_path(cwd, path):
    










    if not os.path.isabs(path):
        path = os.path.join(cwd, path)

    return os.path.abspath(path)
```

## test-9253 — BunchSequences._pad_zeros

- url: https://github.com/zomux/deepy/blob/090fbad22a08a809b12951cd0d4984f5bd432698/deepy/dataset/bunch_seq.py#L66-L72
- repo: zomux/deepy

```python
def _pad_zeros(self, bunch_stack):
        


        min_len = min(map(len, bunch_stack))
        for i in range(len(bunch_stack)):
            bunch_stack[i] = bunch_stack[i][:min_len]
```

## test-283 — SegmentHook.on_error

- url: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/contrib/hooks/segment_hook.py#L83-L90
- repo: apache/airflow

```python
def on_error(self, error, items):
        


        self.log.error('Encountered Segment error: {segment_error} with '
                       'items: {with_items}'.format(segment_error=error,
                                                    with_items=items))
        raise AirflowException('Segment error: {}'.format(error))
```

## test-8863 — Subcmd.lookup

- url: https://github.com/rocky/python3-trepan/blob/14e91bc0acce090d67be145b1ac040cab92ac5f3/trepan/processor/subcmd.py#L28-L36
- repo: rocky/python3-trepan

```python
def lookup(self, subcmd_prefix):
        
        for subcmd_name in list(self.subcmds.keys()):
            if subcmd_name.startswith(subcmd_prefix) \
               and len(subcmd_prefix) >= \
               self.subcmds[subcmd_name].__class__.min_abbrev:
                return self.subcmds[subcmd_name]
            pass
        return None
```

## test-8777 — Session.refresh

- url: https://github.com/josiahcarlson/rom/blob/8b5607a856341df85df33422accc30ba9294dbdb/rom/util.py#L627-L656
- repo: josiahcarlson/rom

```python
def refresh(self, *objects, **kwargs):
        











        self._init()
        from rom import Model
        force = kwargs.get('force')
        for o in objects:
            if isinstance(o, (list, tuple)):
                self.refresh(*o, force=force)
            elif isinstance(o, Model):
                if not o._new:
                    o.refresh(force=force)
                else:
                    # all objects are re-added to the session after refresh,
                    # except for deleted entities...
                    self.add(o)
            else:
                raise ORMError(
                    "Cannot refresh an object that is not an instance of a Model (you provided %r)"%(
                        o,))
```

## test-17445 — filter_glyph_names

- url: https://github.com/aholkner/bacon/blob/edf3810dcb211942d392a8637945871399b0650d/native/Vendor/FreeType/src/tools/glnames.py#L5171-L5183
- repo: aholkner/bacon

```python
def filter_glyph_names( alist, filter ):
  

  count  = 0
  extras = []

  for name in alist:
    try:
      filtered_index = filter.index( name )
    except:
      extras.append( name )

  return extras
```

## test-8050 — _get_file

- url: https://github.com/quantopian/pgcontents/blob/ed36268b7917332d16868208e1e565742a8753e1/pgcontents/query.py#L337-L353
- repo: quantopian/pgcontents

```python
def _get_file(db, user_id, api_path, query_fields, decrypt_func):
    




    result = db.execute(
        _select_file(user_id, api_path, query_fields, limit=1),
    ).first()

    if result is None:
        raise NoSuchFile(api_path)

    if files.c.content in query_fields:
        return to_dict_with_content(query_fields, result, decrypt_func)
    else:
        return to_dict_no_content(query_fields, result)
```

## test-13843 — ControlSuper._calc_dimension

- url: https://github.com/reingart/gui2py/blob/aca0a05f6fcde55c94ad7cc058671a06608b01a4/gui/component.py#L741-L761
- repo: reingart/gui2py

```python
def _calc_dimension(self, dim_val, dim_max, font_dim):
        
        if dim_val is None:
            return -1   # let wx automatic pos/size
        elif isinstance(dim_val, int):
            return dim_val  # use fixed pixel value (absolute)
        elif isinstance(dim_val, basestring):
            if dim_val.endswith("%"):
                # percentaje, relative to parent max size:
                dim_val = int(dim_val[:-1])
                dim_val = dim_val / 100.0 * dim_max
            elif dim_val.endswith("em"):
                # use current font size (suport fractions):
                dim_val = float(dim_val[:-2])
                dim_val = dim_val * font_dim
            elif dim_val.endswith("px"):
                # fixed pixels
                dim_val = dim_val[:-2]
            elif dim_val == "" or dim_val == "auto":
                dim_val = -1
            return int(dim_val)
```

## test-12163 — BaseEvent.remaining_duration

- url: https://github.com/opencast/pyCA/blob/c89b168d4780d157e1b3f7676628c1b131956a88/pyca/db.py#L115-L118
- repo: opencast/pyCA

```python
def remaining_duration(self, time):
        

        return max(0, self.end - max(self.start, time))
```

## test-21330 — IPythonWidget._handle_pyout

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/IPython/frontend/qt/console/ipython_widget.py#L211-L233
- repo: cloud9ers/gurumate

```python
def _handle_pyout(self, msg):
        

        self.log.debug("pyout: %s", msg.get('content', ''))
        if not self._hidden and self._is_from_this_session(msg):
            content = msg['content']
            prompt_number = content.get('execution_count', 0)
            data = content['data']
            if data.has_key('text/html'):
                self._append_plain_text(self.output_sep, True)
                self._append_html(self._make_out_prompt(prompt_number), True)
                html = data['text/html']
                self._append_plain_text('\n', True)
                self._append_html(html + self.output_sep2, True)
            elif data.has_key('text/plain'):
                self._append_plain_text(self.output_sep, True)
                self._append_html(self._make_out_prompt(prompt_number), True)
                text = data['text/plain']
                # If the repr is multiline, make sure we start on a new line,
                # so that its lines are aligned.
                if "\n" in text and not self.output_sep.endswith("\n"):
                    self._append_plain_text('\n', True)
                self._append_plain_text(text + self.output_sep2, True)
```

## test-8742 — wrap_errors

- url: https://github.com/neithere/argh/blob/dcd3253f2994400a6a58a700c118c53765bc50a4/argh/decorators.py#L135-L172
- repo: neithere/argh

```python
def wrap_errors(errors=None, processor=None, *args):
    



























    def wrapper(func):
        if errors:
            setattr(func, ATTR_WRAPPED_EXCEPTIONS, errors)

        if processor:
            setattr(func, ATTR_WRAPPED_EXCEPTIONS_PROCESSOR, processor)

        return func
    return wrapper
```

## test-296 — create_pool

- url: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/www/api/experimental/endpoints.py#L325-L336
- repo: apache/airflow

```python
def create_pool():
    
    params = request.get_json(force=True)
    try:
        pool = pool_api.create_pool(**params)
    except AirflowException as err:
        _log.error(err)
        response = jsonify(error="{}".format(err))
        response.status_code = err.status_code
        return response
    else:
        return jsonify(pool.to_json())
```

## test-16100 — Flask.make_default_options_response

- url: https://github.com/AkihikoITOH/capybara/blob/e86c2173ea386654f4ae061148e8fbe3f25e715c/capybara/virtualenv/lib/python2.7/site-packages/flask/app.py#L1499-L1520
- repo: AkihikoITOH/capybara

```python
def make_default_options_response(self):
        





        adapter = _request_ctx_stack.top.url_adapter
        if hasattr(adapter, 'allowed_methods'):
            methods = adapter.allowed_methods()
        else:
            # fallback for Werkzeug < 0.7
            methods = []
            try:
                adapter.match(method='--')
            except MethodNotAllowed as e:
                methods = e.valid_methods
            except HTTPException as e:
                pass
        rv = self.response_class()
        rv.allow.update(methods)
        return rv
```

## test-14907 — SafeCreationTx.find_valid_random_signature

- url: https://github.com/gnosis/gnosis-py/blob/2a9a5d75a375fc9813ac04df133e6910c82f9d49/gnosis/safe/safe_creation_tx.py#L102-L119
- repo: gnosis/gnosis-py

```python
def find_valid_random_signature(s: int) -> Tuple[int, int]:
        




        for _ in range(10000):
            r = int(os.urandom(31).hex(), 16)
            v = (r % 2) + 27
            if r < secpk1n:
                tx = Transaction(0, 1, 21000, b'', 0, b'', v=v, r=r, s=s)
                try:
                    tx.sender
                    return v, r
                except (InvalidTransaction, ValueError):
                    logger.debug('Cannot find signature with v=%d r=%d s=%d', v, r, s)

        raise ValueError('Valid signature not found with s=%d', s)
```

## test-377 — S3Hook.create_bucket

- url: https://github.com/apache/airflow/blob/b69c686ad8a0c89b9136bb4b31767257eb7b2597/airflow/hooks/S3_hook.py#L72-L90
- repo: apache/airflow

```python
def create_bucket(self, bucket_name, region_name=None):
        







        s3_conn = self.get_conn()
        if not region_name:
            region_name = s3_conn.meta.region_name
        if region_name == 'us-east-1':
            self.get_conn().create_bucket(Bucket=bucket_name)
        else:
            self.get_conn().create_bucket(Bucket=bucket_name,
                                          CreateBucketConfiguration={
                                              'LocationConstraint': region_name
                                          })
```

## test-20109 — normalize_postcode_middleware

- url: https://github.com/arlyon/hyperion/blob/d8de0388ba98b85ce472e0f49ac18fecb14d3343/hyperion/api/util.py#L16-L35
- repo: arlyon/hyperion

```python
async def normalize_postcode_middleware(request, handler):
    


    postcode: Optional[str] = request.match_info.get('postcode', None)

    if postcode is None or postcode == "random":
        return await handler(request)
    elif not is_uk_postcode(postcode):
        raise web.HTTPNotFound(text="Invalid Postcode")

    postcode_processed = postcode.upper().replace(" ", "")
    if postcode_processed == postcode:
        return await handler(request)
    else:
        url_name = request.match_info.route.name
        url = request.app.router[url_name]
        params = dict(request.match_info)
        params['postcode'] = postcode_processed
        raise web.HTTPMovedPermanently(str(url.url_for(**params)))
```

## test-17392 — Route53Connection._send_request

- url: https://github.com/gtaylor/python-route53/blob/b9fc7e258a79551c9ed61e4a71668b7f06f9e774/route53/connection.py#L34-L51
- repo: gtaylor/python-route53

```python
def _send_request(self, path, data, method):
        












        response_body = self._transport.send_request(path, data, method)
        root = etree.fromstring(response_body)
        #print(prettyprint_xml(root))
        return root
```

## test-5379 — ExtendedOptParser.match_time

- url: https://github.com/bloomreach/s4cmd/blob/bb51075bf43703e7cd95aa39288cf7732ec13a6d/s4cmd.py#L1763-L1770
- repo: bloomreach/s4cmd

```python
def match_time(self, value):
    
    m = self.REGEX_TIME.search(value)
    time = datetime.datetime.utcnow().time()
    if m:
      time = datetime.time(int(m.group(1)), int(m.group(2)))
      value = self.REGEX_TIME.sub('', value)
    return (time, value)
```

## test-6522 — BlockRDD._block

- url: https://github.com/lensacom/sparkit-learn/blob/0498502107c1f7dcf33cda0cdb6f5ba4b42524b7/splearn/rdd.py#L132-L148
- repo: lensacom/sparkit-learn

```python
def _block(self, rdd, bsize, dtype):
        













        return rdd.mapPartitions(
            lambda x: _block_collection(x, dtype, bsize))
```

## test-14863 — ThumbnailManager.get

- url: https://github.com/ui/django-thumbnails/blob/5cef55e7f167060458709ed760dd43981124796a/thumbnails/files.py#L69-L91
- repo: ui/django-thumbnails

```python
def get(self, size, create=True):
        






        if self._thumbnails is None:
            self._refresh_cache()

        thumbnail = self._thumbnails.get(size)

        if thumbnail is None:
            thumbnail = images.get(self.source_image.name, size,
                                   self.metadata_backend, self.storage)

            if thumbnail is None:
                thumbnail = self.create(size)

            self._thumbnails[size] = thumbnail

        return thumbnail
```

## test-15282 — splat

- url: https://github.com/drhagen/parsita/blob/d97414a05541f48231381f607d1d2e6b50781d39/parsita/util.py#L22-L44
- repo: drhagen/parsita

```python
def splat(f: Callable[..., A]) -> Callable[[Iterable], A]:
    

















    def splatted(args):
        return f(*args)

    return splatted
```

## test-12158 — run_subprocess

- url: https://github.com/openergy/oplus/blob/f095868d1990c1d126e906ada6acbab26348b3d3/oplus/util.py#L81-L116
- repo: openergy/oplus

```python
def run_subprocess(command, cwd=None, stdout=None, stderr=None, shell=False, beat_freq=None):
    









    sys.encoding = CONF.encoding
    # prepare variables
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr

    # run subprocess
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        shell=shell,
        universal_newlines=True
    ) as sub_p:
        # link output streams
        with redirect_stream(sub_p.stdout, stdout), redirect_stream(sub_p.stderr, stderr):
            while True:
                try:
                    sub_p.wait(timeout=beat_freq)
                    break
                except subprocess.TimeoutExpired:
                    stdout.write("subprocess is still running\n")
                    if hasattr(sys.stdout, "flush"):
                        sys.stdout.flush()
        return sub_p.returncode
```

## test-5474 — ImportsChecker._report_external_dependencies

- url: https://github.com/PyCQA/pylint/blob/2bf5c61a3ff6ae90613b81679de42c0f19aea600/pylint/checkers/imports.py#L903-L909
- repo: PyCQA/pylint

```python
def _report_external_dependencies(self, sect, _, _dummy):
        
        dep_info = _make_tree_defs(self._external_dependencies_info().items())
        if not dep_info:
            raise EmptyReportError()
        tree_str = _repr_tree_defs(dep_info)
        sect.append(VerbatimText(tree_str))
```

## test-19522 — GChart.scale

- url: https://github.com/appknox/google-chartwrapper/blob/3769aecbef6c83b6cd93ee72ece478ffe433ac57/GChartWrapper/GChart.py#L271-L283
- repo: appknox/google-chartwrapper

```python
def scale(self, *args):
        









        self._scale =  [','.join(map(smart_str, args))]
        return self
```

## test-6010 — Context._check_env_vars_set

- url: https://github.com/pyca/pyopenssl/blob/1fbe064c50fd030948141d7d630673761525b0d0/src/OpenSSL/SSL.py#L861-L870
- repo: pyca/pyopenssl

```python
def _check_env_vars_set(self, dir_env_var, file_env_var):
        




        return (
            os.environ.get(file_env_var) is not None or
            os.environ.get(dir_env_var) is not None
        )
```

## test-5011 — MBox._copy_mbox

- url: https://github.com/chaoss/grimoirelab-perceval/blob/41c908605e88b7ebc3a536c643fa0f212eaf9e0e/perceval/backends/core/mbox.py#L238-L247
- repo: chaoss/grimoirelab-perceval

```python
def _copy_mbox(self, mbox):
        

        tmp_path = tempfile.mktemp(prefix='perceval_')

        with mbox.container as f_in:
            with open(tmp_path, mode='wb') as f_out:
                for l in f_in:
                    f_out.write(l)
        return tmp_path
```

## test-7276 — Datasets.add_from_raw_data

- url: https://github.com/Azure/Azure-MachineLearning-ClientLibrary-Python/blob/d1211b289747671898eb063013e0dc53d3c80acd/azureml/__init__.py#L509-L545
- repo: Azure/Azure-MachineLearning-ClientLibrary-Python

```python
def add_from_raw_data(self, raw_data, data_type_id, name, description):
        





























        _not_none('raw_data', raw_data)
        _not_none_or_empty('data_type_id', data_type_id)
        _not_none_or_empty('name', name)
        _not_none_or_empty('description', description)

        return self._upload(raw_data, data_type_id, name, description)
```

## test-19245 — Encoder._encode_chunk

- url: https://github.com/thusoy/pwm/blob/fff7d755c34f3a7235a8bf217ffa2ff5aed4926f/pwm/encoding.py#L55-L61
- repo: thusoy/pwm

```python
def _encode_chunk(self, data, index):
        



        chunk = self._get_chunk(data, index)
        return self._encode_long(self._chunk_to_long(chunk))
```

## test-1326 — embed_check_nonnegative_integer_form

- url: https://github.com/tensorflow/probability/blob/e87fe34111d68c35db0f9eeb4935f1ece9e1a8f5/tensorflow_probability/python/internal/distribution_util.py#L692-L707
- repo: tensorflow/probability

```python
def embed_check_nonnegative_integer_form(
    x, name="embed_check_nonnegative_integer_form"):
  
  with tf.name_scope(name):
    x = tf.convert_to_tensor(value=x, name="x")
    assertions = [
        assert_util.assert_non_negative(
            x, message="'{}' must be non-negative.".format(x)),
    ]
    if not dtype_util.is_integer(x.dtype):
      assertions += [
          assert_integer_form(
              x,
              message="'{}' cannot contain fractional components.".format(x)),
      ]
    return with_dependencies(assertions, x)
```

## test-8072 — to_dict_with_content

- url: https://github.com/quantopian/pgcontents/blob/ed36268b7917332d16868208e1e565742a8753e1/pgcontents/db_utils.py#L84-L101
- repo: quantopian/pgcontents

```python
def to_dict_with_content(fields, row, decrypt_func):
    








    assert(len(fields) == len(row))

    field_names = list(map(_get_name, fields))
    assert 'content' in field_names, "Missing content field."

    result = dict(zip(field_names, row))
    result['content'] = decrypt_func(result['content'])
    return result
```

## test-20776 — dir2

- url: https://github.com/cloud9ers/gurumate/blob/075dc74d1ee62a8c6b7a8bf2b271364f01629d1e/environment/lib/python2.7/site-packages/IPython/utils/dir2.py#L34-L73
- repo: cloud9ers/gurumate

```python
def dir2(obj):
    











    # Start building the attribute list via dir(), and then complete it
    # with a few extra special-purpose calls.

    words = set(dir(obj))

    if hasattr(obj, '__class__'):
        #words.add('__class__')
        words |= set(get_class_members(obj.__class__))


    # for objects with Enthought's traits, add trait_names() list
    # for PyCrust-style, add _getAttributeNames() magic method list
    for attr in ('trait_names', '_getAttributeNames'):
        if hasattr(obj, attr):
            try:
                func = getattr(obj, attr)
                if callable(func):
                    words |= set(func())
            except:
                # TypeError: obj is class not instance
                pass

    # filter out non-string attributes which may be stuffed by dir() calls
    # and poor coding in third-party modules

    words = [w for w in words if isinstance(w, basestring)]
    return sorted(words)
```

## test-4656 — MinHashLSH.get_subset_counts

- url: https://github.com/ekzhu/datasketch/blob/b3e4129987890a2beb04f2c0b6dc618ae35f2e14/datasketch/lsh.py#L268-L287
- repo: ekzhu/datasketch

```python
def get_subset_counts(self, *keys):
        







        if self.prepickle:
            key_set = [pickle.dumps(key) for key in set(keys)]
        else:
            key_set = list(set(keys))
        hashtables = [unordered_storage({'type': 'dict'}) for _ in
                      range(self.b)]
        Hss = self.keys.getmany(*key_set)
        for key, Hs in zip(key_set, Hss):
            for H, hashtable in zip(Hs, hashtables):
                hashtable.insert(H, key)
        return [hashtable.itemcounts() for hashtable in hashtables]
```

## test-10221 — s3errors

- url: https://github.com/PyFilesystem/s3fs/blob/1c5e3a1b6abbb9dff91ea7fc4cec7353798cd536/fs_s3fs/_s3fs.py#L171-L192
- repo: PyFilesystem/s3fs

```python
def s3errors(path):
    
    try:
        yield
    except ClientError as error:
        _error = error.response.get("Error", {})
        error_code = _error.get("Code", None)
        response_meta = error.response.get("ResponseMetadata", {})
        http_status = response_meta.get("HTTPStatusCode", 200)
        error_msg = _error.get("Message", None)
        if error_code == "NoSuchBucket":
            raise errors.ResourceError(path, exc=error, msg=error_msg)
        if http_status == 404:
            raise errors.ResourceNotFound(path)
        elif http_status == 403:
            raise errors.PermissionDenied(path=path, msg=error_msg)
        else:
            raise errors.OperationFailed(path=path, exc=error)
    except SSLError as error:
        raise errors.OperationFailed(path, exc=error)
    except EndpointConnectionError as error:
        raise errors.RemoteConnectionError(path, exc=error, msg="{}".format(error))
```

## test-2932 — WebTV._get_streams

- url: https://github.com/streamlink/streamlink/blob/c8ed1daff14ac03195870238b9b900c1109dd5c1/src/streamlink/plugins/webtv.py#L49-L76
- repo: streamlink/streamlink

```python
def _get_streams(self):
        



        headers = {}
        res = self.session.http.get(self.url, headers=headers)
        headers["Referer"] = self.url

        sources = self._sources_re.findall(res.text)
        if len(sources):
            sdata = parse_json(sources[0], schema=self._sources_schema)
            for source in sdata:
                self.logger.debug("Found stream of type: {}", source[u'type'])
                if source[u'type'] == u"application/vnd.apple.mpegurl":
                    url = update_scheme(self.url, source[u"src"])

                    try:
                        # try to parse the stream as a variant playlist
                        variant = HLSStream.parse_variant_playlist(self.session, url, headers=headers)
                        if variant:
                            for q, s in variant.items():
                                yield q, s
                        else:
                            # and if that fails, try it as a plain HLS stream
                            yield 'live', HLSStream(self.session, url, headers=headers)
                    except IOError:
                        self.logger.warning("Could not open the stream, perhaps the channel is offline")
```

## test-6967 — Inactive._backup

- url: https://github.com/funilrys/PyFunceble/blob/cdf69cbde120199171f7158e1c33635753e6e2f5/PyFunceble/database.py#L270-L279
- repo: funilrys/PyFunceble

```python
def _backup(self):
        



        if PyFunceble.CONFIGURATION["inactive_database"]:
            # The database subsystem is activated.

            # We save the current database state into the database file.
            Dict(PyFunceble.INTERN["inactive_db"]).to_json(self.inactive_db_path)
```

## test-14083 — Provider._make_json_response

- url: https://github.com/NateFerrero/oauth2lib/blob/d161b010f8a596826050a09e5e94d59443cc12d9/oauth2lib/provider.py#L61-L80
- repo: NateFerrero/oauth2lib

```python
def _make_json_response(self, data, headers=None, status_code=200):
        









        response_headers = {}
        if headers is not None:
            response_headers.update(headers)
        response_headers['Content-Type'] = 'application/json;charset=UTF-8'
        response_headers['Cache-Control'] = 'no-store'
        response_headers['Pragma'] = 'no-cache'
        return self._make_response(json.dumps(data),
                                   response_headers,
                                   status_code)
```

## test-1750 — Trie._get_letters

- url: https://github.com/deepmipt/DeepPavlov/blob/f3e4a69a3764d25d2f5bad4f1f1aebc872b00f9c/deeppavlov/models/spelling_correction/levenshtein/tabled_trie.py#L271-L282
- repo: deepmipt/DeepPavlov

```python
def _get_letters(self, index, return_indexes=False):
        


        if self.dict_storage:
            answer = list(self.graph[index].keys())
        else:
            answer =  [i for i, elem in enumerate(self.graph[index])
                       if elem != Trie.NO_NODE]
        if not return_indexes:
            answer = [(self.alphabet[i] if i >= 0 else " ") for i in answer]
        return answer
```

## test-6695 — fdr

- url: https://github.com/neurosynth/neurosynth/blob/948ce7edce15d7df693446e76834e0c23bfe8f11/neurosynth/analysis/stats.py#L62-L69
- repo: neurosynth/neurosynth

```python
def fdr(p, q=.05):
    

    s = np.sort(p)
    nvox = p.shape[0]
    null = np.array(range(1, nvox + 1), dtype='float') * q / nvox
    below = np.where(s <= null)[0]
    return s[max(below)] if len(below) else -1
```

## test-16127 — UninstallPathSet.commit

- url: https://github.com/AkihikoITOH/capybara/blob/e86c2173ea386654f4ae061148e8fbe3f25e715c/capybara/virtualenv/lib/python2.7/site-packages/pip/req/req_uninstall.py#L150-L155
- repo: AkihikoITOH/capybara

```python
def commit(self):
        
        if self.save_dir is not None:
            rmtree(self.save_dir)
            self.save_dir = None
            self._moved_paths = []
```

## test-6817 — Player.seek

- url: https://github.com/mental32/spotify.py/blob/bb296cac7c3dd289908906b7069bd80f43950515/spotify/models/player.py#L96-L109
- repo: mental32/spotify.py

```python
async def seek(self, pos, *, device: Optional[SomeDevice] = None):
        











        await self._user.http.seek_playback(pos, device_id=str(device))
```

## test-10171 — get

- url: https://github.com/singularityhub/sregistry-cli/blob/abc96140a1d15b5e96d83432e1e0e1f4f8f36331/sregistry/database/sqlite.py#L76-L110
- repo: singularityhub/sregistry-cli

```python
def get(self, name, quiet=False):
    







    from sregistry.database.models import Collection, Container
    names = parse_image_name( remove_uri (name) )
 
    # First look for a collection (required)
    collection = self.get_collection(name=names['collection'])
    container = None

    if collection is not None:
        container = self.get_container(collection_id=collection.id,
                                       name=names['image'], 
                                       tag=names['tag'],
                                       version=names['version'])

        if container is not None and quiet is False:

            # The container image file exists [local]
            if container.image is not None:
                print(container.image)

            # The container has a url (but not local file)
            elif container.url is not None:
                print(container.url)
            else:
                bot.info('No storage file found for %s' %name)

    return container
```

## test-9868 — to_underscore

- url: https://github.com/slacy/minimongo/blob/29f38994831163b17bc625c82258068f1f90efa5/minimongo/model.py#L241-L249
- repo: slacy/minimongo

```python
def to_underscore(string):
    




    new_string = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', string)
    new_string = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', new_string)
    return new_string.lower()
```

## test-3480 — track_parallel_progress

- url: https://github.com/open-mmlab/mmcv/blob/0d77f61450aab4dde8b8585a577cc496acb95d7f/mmcv/utils/progressbar.py#L108-L174
- repo: open-mmlab/mmcv

```python
def track_parallel_progress(func,
                            tasks,
                            nproc,
                            initializer=None,
                            initargs=None,
                            bar_width=50,
                            chunksize=1,
                            skip_first=False,
                            keep_order=True):
    
























    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], collections_abc.Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, collections_abc.Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(task_num, bar_width, start)
    results = []
    if keep_order:
        gen = pool.imap(func, tasks, chunksize)
    else:
        gen = pool.imap_unordered(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    sys.stdout.write('\n')
    pool.close()
    pool.join()
    return results
```

## test-1612 — _prob_chain_rule_flatten

- url: https://github.com/tensorflow/probability/blob/e87fe34111d68c35db0f9eeb4935f1ece9e1a8f5/tensorflow_probability/python/distributions/joint_distribution_named.py#L245-L267
- repo: tensorflow/probability

```python
def _prob_chain_rule_flatten(named_makers):
  
  def _make(dist_fn, args):
    if args is None:
      return lambda *_: dist_fn
    if not args:
      return lambda *_: dist_fn()
    def _fn(*xs):
      kwargs = dict(zip(args, reversed(xs[-len(args):])))
      kwargs.pop('_', None)
      return dist_fn(**kwargs)
    return _fn
  named_makers = _convert_to_dict(named_makers)
  g = {k: (None if distribution_util.is_distribution_instance(v)
           else joint_distribution_sequential._get_required_args(v))  # pylint: disable=protected-access
       for k, v in named_makers.items()}
  g = _best_order(g)
  dist_fn_name, dist_fn_args = zip(*g)
  dist_fn_args = tuple(None if a is None else tuple(a) for a in dist_fn_args)
  dist_fn_wrapped = tuple(_make(named_makers[name], parents)
                          for (name, parents) in g)
  dist_fn = tuple(named_makers.get(n) for n in dist_fn_name)
  return dist_fn, dist_fn_wrapped, dist_fn_args, dist_fn_name
```

## test-14605 — CryptFileKeyring._check_scheme

- url: https://github.com/frispete/keyrings.cryptfile/blob/cfa80d4848a5c3c0aeee41a954b2b120c80e69b2/keyrings/cryptfile/cryptfile.py#L132-L162
- repo: frispete/keyrings.cryptfile

```python
def _check_scheme(self, config):
        





        try:
            scheme = config.get(
                escape_for_ini('keyring-setting'),
                escape_for_ini('scheme'),
            )
        except (configparser.NoSectionError, configparser.NoOptionError):
            raise AttributeError("Encryption scheme missing")

        # extract AES mode
        aesmode = scheme[-3:]
        if aesmode not in self._get_mode():
            raise ValueError("Encryption scheme invalid: %s" % (aesmode))

        # setup AES mode
        self.aesmode = aesmode

        # remove pointless crypto module name
        if scheme.startswith('PyCryptodome '):
            scheme = scheme[13:]

        # check other scheme properties
        if scheme != self.scheme:
            raise ValueError("Encryption scheme mismatch "
                             "(exp.: %s, found: %s)" % (self.scheme, scheme))
```

## test-15465 — SASLStateMachine.response

- url: https://github.com/horazont/aiosasl/blob/af58bf30f688757e58af6e87892d35a8ce798482/aiosasl/__init__.py#L464-L509
- repo: horazont/aiosasl

```python
def response(self, payload):
        







        if self._state == SASLState.SUCCESS_SIMULATE_CHALLENGE:
            if payload != b"":
                # XXX: either our mechanism is buggy or the server
                # sent SASLState.SUCCESS before all challenge-response
                # messages defined by the mechanism were sent
                self._state = SASLState.FAILURE
                raise SASLFailure(
                    None,
                    "protocol violation: mechanism did not"
                    " respond with an empty response to a"
                    " challenge with final data – this suggests"
                    " a protocol-violating early success from the server."
                )
            self._state = SASLState.SUCCESS
            return SASLState.SUCCESS, None

        if self._state != SASLState.CHALLENGE:
            raise RuntimeError(
                "no challenge has been made or negotiation failed")

        try:
            next_state, payload = yield from self.interface.respond(payload)
        except SASLFailure:
            self._state = SASLState.FAILURE
            raise

        next_state = SASLState.from_reply(next_state)

        # unfold the (SASLState.SUCCESS, payload) to a sequence of
        # (SASLState.CHALLENGE, payload), (SASLState.SUCCESS, None) for the SASLMethod
        # to allow uniform treatment of both cases
        if next_state == SASLState.SUCCESS and payload is not None:
            self._state = SASLState.SUCCESS_SIMULATE_CHALLENGE
            return SASLState.CHALLENGE, payload

        self._state = next_state
        return next_state, payload
```

## test-3579 — BotState.save_changes

- url: https://github.com/Microsoft/botbuilder-python/blob/274663dd91c811bae6ac4488915ba5880771b0a7/libraries/botbuilder-core/botbuilder/core/bot_state.py#L89-L104
- repo: Microsoft/botbuilder-python

```python
async def save_changes(self, turn_context: TurnContext, force: bool = False) -> None:
        




        if turn_context == None:
            raise TypeError('BotState.save_changes(): turn_context cannot be None.')
        
        cached_state = turn_context.turn_state.get(self._context_service_key)
        
        if force or (cached_state != None and cached_state.is_changed == True):
            storage_key = self.get_storage_key(turn_context)
            changes : Dict[str, object] = { storage_key: cached_state.state }
            await self._storage.write(changes)
            cached_state.hash = cached_state.compute_hash(cached_state.state)
```

## test-4989 — GroupsioClient.__find_group_id

- url: https://github.com/chaoss/grimoirelab-perceval/blob/41c908605e88b7ebc3a536c643fa0f212eaf9e0e/perceval/backends/core/groupsio.py#L229-L240
- repo: chaoss/grimoirelab-perceval

```python
def __find_group_id(self):
        

        group_subscriptions = self.subscriptions(self.auth)

        for subscriptions in group_subscriptions:
            for sub in subscriptions:
                if sub['group_name'] == self.group_name:
                    return sub['group_id']

        msg = "Group id not found for group name %s" % self.group_name
        raise BackendError(cause=msg)
```
