{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.trials.last;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils
  , SDL2
  , sdl.app.trials
  , sdl.app.stimuli.contract
  , sdl.app.stimuli
  , sdl.app.stimuli.last
  ;

type

  { TLastTrial }

  TLastTrial = class sealed (TTrial)
    private
      FStimuli : TStimuli;
    protected
      function GetIStimuli: IStimuli; override;
      //function Header : string; override;
    public
      constructor Create; override;
      destructor Destroy; override;
  end;


implementation

{ TLastTrial }

function TLastTrial.GetIStimuli: IStimuli;
begin
  Result := FStimuli.AsIStimuli;
end;

//function TLastTrial.Header: string;
//begin
//  Result := 'Response';
//end;

constructor TLastTrial.Create;
begin
  inherited Create;
  FStimuli := TLastStimuli.Create;
  FStimuli.Trial := Self as TObject;
  FStimuli.OnFinalize := nil;
end;

destructor TLastTrial.Destroy;
begin
  FStimuli.Free;
  inherited Destroy;
end;

end.

