import React, {
  Component, Fragment
} from 'react';

import moment from 'moment';

import {
  EuiButton,
  EuiButtonEmpty,
  EuiCode,
  EuiCodeBlock,
  EuiDatePicker,
  EuiFieldText,
  EuiFlexGroup,
  EuiFlexItem,
  EuiForm,
  EuiFormControlLayout,
  EuiFormLabel,
  EuiFormRow,
  EuiHeader,
  EuiHeaderBreadcrumb,
  EuiHeaderBreadcrumbs,
  EuiHeaderSection,
  EuiHeaderSectionItem,
  EuiHeaderSectionItemButton,
  EuiHeaderLogo,
  EuiImage,
  EuiIcon,
  EuiModal,
  EuiModalHeader,
  EuiModalHeaderTitle,
  EuiModalBody,
  EuiModalFooter,
  EuiPage,
  EuiPageBody,
  EuiPageContent,
  EuiPageContentBody,
  EuiPageContentHeader,
  EuiPageHeader,
  EuiPanel,
  EuiSelect,
  EuiSpacer,
  EuiText,
  EuiTextArea,
  EuiTitle,
} from '@elastic/eui';

import IndexSelectionFormGroup from './IndexSelectionFormGroup'

import {
  loadIndices,
  submitComment,
} from '../lib/esClient'


export default class NewCommentModal extends Component {

  constructor(props) {
    super(props);

    this.state = {
      startDate: null,
      commentValue: null,
      urlValue: null,
      selectedIndex: null,
      errors: {
        startDate: [],
        commentValue: [],
        urlValue: []
      }
    };

    this.closeModal = props.onClose;
    this.addToast = props.addToast;

    this.handleIndexChange   = this.handleIndexChange.bind(this);
    this.handleDateChange    = this.handleDateChange.bind(this);
    this.handleCommentChange = this.handleCommentChange.bind(this);
    this.handleUrlChange     = this.handleUrlChange.bind(this);
    this.submit              = this.submit.bind(this);
  }

  handleIndexChange = selectedIndex => {
    this.setState({ selectedIndex });
  }

  handleDateChange = startDate => {

    const { errors } = this.state;

    this.setState({
      startDate,
      errors: {...errors, startDate:[]}
    });
  }

  handleCommentChange = e => {

    const { errors } = this.state;

    this.setState({
      commentValue: e.target.value,
      errors: {...errors, commentValue:[]}
    });
  }

  handleUrlChange = e => {

    const { errors } = this.state;

    this.setState({
      urlValue: e.target.value,
    });
  }

  submit() {

    // validate data
    let errors = {};

    if (!this.state.startDate)
      errors['startDate'] = ['日時は必須です'];
    else if (!moment(this.state.startDate).isValid())
      errors['startDate'] = ['フォーマットが不正です'];

    if (!this.state.commentValue)
      errors['commentValue'] = ['コメントは必須です'];

    if (Object.keys(errors).length) {

      // fill other props to explicit empty value
      errors = {
        startDate: [],
        commentValue: [],
        ...errors
      }

      this.setState({errors});
      return;
    }

    // submit to ES

    submitComment(this.state.selectedIndex, this.state.startDate, this.state.commentValue, this.state.urlValue)
      .then((res) => {

        if (!res.status) {
          // error loading indices
          this.addToast({
            title: "コメント追加エラー",
            type: "danger",
            msg: (<EuiCodeBlock language="json">{JSON.stringify(res.response.data)}</EuiCodeBlock>)
          });

          return;
        }

        this.addToast({
          title: "新しいコメントが追加されました",
          type: "success"
        });

        // reset state
        this.setState({
          commentValue: '',
        });

        this.setState({
          urlValue: '',
        })

        this.closeModal();
      })
      .catch((err) => {console.log(err)})
  }


  render() {

    return(

      <EuiModal
        onClose={this.closeModal}
        style={{ width: '800px' }}
      >
        <EuiModalHeader>
          <EuiModalHeaderTitle >
            コメントを追加する
          </EuiModalHeaderTitle>
        </EuiModalHeader>

        <EuiModalBody>
          <EuiFlexGroup>
            <EuiFlexItem>
              <EuiFormRow
                label="日時"
                isInvalid={!!this.state.errors.startDate.length}
                error={this.state.errors.startDate}
                fullWidth
              >
                <EuiDatePicker
                  showTimeSelect
                  isInvalid={!!this.state.errors.startDate.length}
                  selected={this.state.startDate}
                  onChange={this.handleDateChange}
                  dateFormat='DD/MM/YYYY HH:mm'
                  placeholder="コメントを追加する日時を選択してください"
                  fullWidth
                />
              </EuiFormRow>
              {/*
              <EuiFormLabel>set to now</EuiFormLabel>
              */}
            </EuiFlexItem>
          </EuiFlexGroup>

          <EuiSpacer />

          <IndexSelectionFormGroup onChange={this.handleIndexChange} addToast={this.addToast} />

          <EuiSpacer />

          <EuiFlexGroup>
            <EuiFlexItem>
              <EuiFormRow
                label="コメント"
                isInvalid={!!this.state.errors.commentValue.length}
                error={this.state.errors.commentValue}
                fullWidth
                >
                <EuiTextArea
                  isInvalid={!!this.state.errors.commentValue.length}
                  onChange={this.handleCommentChange}
                  placeholder="コメントを記述してください"
                  fullWidth
                />
              </EuiFormRow>
            </EuiFlexItem>
          </EuiFlexGroup>
          <EuiFlexGroup>
            <EuiFlexItem>
              <EuiFormRow label="URL" fullWidth>
                <EuiFieldText
                  onChange={this.handleUrlChange}
                  placeholder="検索結果や対象となるログのURLを入力してください"
                  fullWidth
                />
              </EuiFormRow>
            </EuiFlexItem>
          </EuiFlexGroup>




        </EuiModalBody>

        <EuiModalFooter>
          <EuiButtonEmpty onClick={this.closeModal}>
            キャンセル
          </EuiButtonEmpty>

          <EuiButton onClick={this.submit} fill>
            追加
          </EuiButton>
        </EuiModalFooter>

      </EuiModal>
    );
  }

};
