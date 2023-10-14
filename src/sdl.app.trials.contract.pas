{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.trials.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses Classes, Session.Configuration, sdl.app.stimuli.contract;

type

  { ITrial }

  ITrial = interface
    ['{3652F678-82C6-4FDC-AC8B-E41CF4453138}']
    function ConsequenceDelay: Cardinal;
    function ConsequenceInterval: Cardinal;
    function GetOnTrialEnd: TNotifyEvent;
    function GetTrialData: TTrialData;
    function InterTrialInterval : Cardinal;
    function AsITrial : ITrial;
    function GetIStimuli : IStimuli;
    procedure Hide;
    procedure SetOnTrialEnd(ANotifyEvent: TNotifyEvent);
    procedure SetTrialData(TrialData: TTrialData);
    procedure Show;
    procedure EndTrial;
    property Data : TTrialData read GetTrialData write SetTrialData;
    property OnTrialEnd : TNotifyEvent read GetOnTrialEnd write SetOnTrialEnd;
  end;

implementation

end.

