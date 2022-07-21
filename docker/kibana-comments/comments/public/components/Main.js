import React, {
  Component, Fragment
} from 'react';

import '@elastic/eui/dist/eui_theme_light.css';

import {
  EuiFlexItem,
  EuiFlexGroup,
  EuiHeader,
  EuiHeaderSection,
  EuiHeaderSectionItem,
  EuiIcon,
  EuiPage,
  EuiPageBody,
  EuiPageContent,
  EuiPageContentBody,
  EuiPageHeader,
  EuiSpacer,
  EuiText,
  EuiTitle,
  EuiImage,
} from '@elastic/eui';

import logoUrl from "../logo.png";

import ListComments from '../components/ListComments'


const Main = (props) => {



  return(
    <Fragment>
    <EuiPage>
      <EuiPageBody>
        <EuiPageHeader>

          <EuiHeader style={{width: "100%"}}>
            <EuiHeaderSection>
              <EuiHeaderSectionItem>
	        <EuiImage size="m" hasShadow alt="logo" url={logoUrl} />
              </EuiHeaderSectionItem>
            </EuiHeaderSection>
          </EuiHeader>
        </EuiPageHeader>

        <EuiSpacer />

        <EuiPageContent>


          <EuiPageContentBody>
            <ListComments />
          </EuiPageContentBody>
        </EuiPageContent>
      </EuiPageBody>

      <EuiSpacer />

   </EuiPage>
   </Fragment>
  );

};

export default Main
